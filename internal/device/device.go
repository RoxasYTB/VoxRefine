// Package device connects kyone's real-time enhancer to the sound card: it
// captures a microphone, hands blocks to a processing function and plays the
// result on another device, typically a virtual cable that chat applications
// then use as their microphone. The audio backend is miniaudio (WASAPI on
// Windows, Core Audio on macOS, ALSA/Pulse on Linux) through cgo; without cgo
// a stub reports that live mode is unavailable.
package device

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
)

// SampleRate is the only rate the live engine runs at; the backend converts
// from and to whatever the devices use natively.
const SampleRate = 48000

// Kind selects capture (microphone) or playback devices.
type Kind int

const (
	Capture Kind = iota
	Playback
)

func (k Kind) String() string {
	if k == Capture {
		return "capture"
	}
	return "playback"
}

// Info describes one audio device.
type Info struct {
	Name    string
	Default bool
}

// ErrNoBackend is returned when this build has no audio backend (cgo off).
var ErrNoBackend = errors.New("live mode needs an audio backend: rebuild kyone with cgo enabled (a C compiler is required)")

// Config selects the devices and the buffering.
type Config struct {
	// Input and Output are case-insensitive substrings of device names. Empty
	// means the system default; for Output, see VirtualCable.
	Input, Output string
	// BlockMS is the length of one audio period in ms (default 10).
	BlockMS int
}

// Stats are live counters, safe to read while running.
type Stats struct {
	Underruns, Overruns uint64
	Frames              uint64
	InputPeak           float32 // peak of the last second of input, 0..1
	OutputPeak          float32
}

// virtualCableHints are name fragments of the loopback devices that chat
// applications can select as a microphone.
var virtualCableHints = []string{"blackhole", "cable input", "vb-audio", "voicemeeter", "loopback audio", "vac "}

// IsVirtualCable reports whether a playback device name looks like a virtual
// audio cable.
func IsVirtualCable(name string) bool {
	n := strings.ToLower(name)
	for _, h := range virtualCableHints {
		if strings.Contains(n, h) {
			return true
		}
	}
	return false
}

// VirtualCable returns the first playback device that looks like a virtual
// cable, or "" if none is installed.
func VirtualCable(playback []Info) string {
	for _, d := range playback {
		if IsVirtualCable(d.Name) {
			return d.Name
		}
	}
	return ""
}

// InstallHint tells the user how to get a virtual cable on this platform.
func InstallHint() string {
	switch runtime.GOOS {
	case "darwin":
		return "install BlackHole (free): brew install blackhole-2ch, then choose it as the microphone in your chat application"
	case "windows":
		return "install VB-CABLE (free): https://vb-audio.com/Cable/ — then choose \"CABLE Output\" as the microphone in your chat application"
	default:
		return "create a virtual sink, e.g. with PipeWire/PulseAudio (module-null-sink), and choose its monitor as the microphone"
	}
}

// match finds the first device whose name contains want (case-insensitive).
// An empty want selects the default device, reported as index -1.
func match(devices []Info, want string, kind Kind) (int, error) {
	if want == "" {
		return -1, nil
	}
	w := strings.ToLower(want)
	for i, d := range devices {
		if strings.Contains(strings.ToLower(d.Name), w) {
			return i, nil
		}
	}
	names := make([]string, len(devices))
	for i, d := range devices {
		names[i] = fmt.Sprintf("%q", d.Name)
	}
	return 0, fmt.Errorf("no %s device matches %q (available: %s)", kind, want, strings.Join(names, ", "))
}

// ring is a small mutex-protected sample FIFO bridging the two device
// callbacks, which run on different threads and clocks.
type ring struct {
	mu         sync.Mutex
	buf        []float32
	head, size int
}

func newRing(capacity int) *ring { return &ring{buf: make([]float32, capacity)} }

func (r *ring) len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.size
}

// push appends x; if the ring is full the oldest samples are dropped and the
// number dropped is returned.
func (r *ring) push(x []float32) (dropped int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, v := range x {
		if r.size == len(r.buf) {
			r.head = (r.head + 1) % len(r.buf)
			r.size--
			dropped++
		}
		r.buf[(r.head+r.size)%len(r.buf)] = v
		r.size++
	}
	return dropped
}

// dropOldest discards samples until at most keep remain; returns how many.
func (r *ring) dropOldest(keep int) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for r.size > keep {
		r.head = (r.head + 1) % len(r.buf)
		r.size--
		n++
	}
	return n
}

// pop fills dst with up to len(dst) samples and returns how many it got.
func (r *ring) pop(dst []float32) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := min(len(dst), r.size)
	for i := 0; i < n; i++ {
		dst[i] = r.buf[(r.head+i)%len(r.buf)]
	}
	r.head = (r.head + n) % len(r.buf)
	r.size -= n
	return n
}

// peakMeter tracks the largest absolute sample, resettable by the reader.
type peakMeter struct{ bits atomic.Uint32 }

func (p *peakMeter) observe(x []float32) {
	var m float32
	for _, v := range x {
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	for {
		old := p.bits.Load()
		if m <= float32FromBits(old) || p.bits.CompareAndSwap(old, float32Bits(m)) {
			return
		}
	}
}

func (p *peakMeter) take() float32 { return float32FromBits(p.bits.Swap(0)) }
