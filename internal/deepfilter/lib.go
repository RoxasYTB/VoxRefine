package deepfilter

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Lib is the DeepFilterNet C library (libdf) loaded at run time, without cgo,
// through purego. Build it from the DeepFilterNet repository:
//
//	cargo build --release -p deep_filter --lib --features capi
//
// which produces libdf.dylib (macOS), df.dll (Windows) or libdf.so (Linux).
type Lib struct {
	handle uintptr

	create      func(path string, attenLim float32, logLevel uintptr) uintptr
	free        func(state uintptr)
	frameLength func(state uintptr) uintptr
	process     func(state uintptr, in, out unsafe.Pointer) float32
	setAtten    func(state uintptr, limDB float32)
	setPF       func(state uintptr, beta float32)
}

// FindLib locates libdf: the explicit path, then KYONE_DF_LIB, then next to
// kyone.
func FindLib(explicit string) (string, error) {
	for _, p := range []string{explicit, os.Getenv("KYONE_DF_LIB")} {
		if p != "" {
			if _, err := os.Stat(p); err != nil {
				return "", fmt.Errorf("DeepFilterNet library not found at %q", p)
			}
			return p, nil
		}
	}
	if p := firstExisting(LibName()); p != "" {
		return p, nil
	}
	return "", fmt.Errorf("DeepFilterNet library (%s) not found: build it with `cargo build --release -p deep_filter --lib --features capi` in a clone of github.com/Rikorose/DeepFilterNet, then put it next to kyone or pass --deepfilter-lib", LibName())
}

// FindModel locates the model archive: the explicit path, then KYONE_DF_MODEL,
// then DefaultModel next to kyone.
func FindModel(explicit string) (string, error) {
	for _, p := range []string{explicit, os.Getenv("KYONE_DF_MODEL")} {
		if p != "" {
			if _, err := os.Stat(p); err != nil {
				return "", fmt.Errorf("DeepFilterNet model not found at %q", p)
			}
			return p, nil
		}
	}
	if p := firstExisting(DefaultModel); p != "" {
		return p, nil
	}
	return "", fmt.Errorf("DeepFilterNet model not found: copy models/%s from the DeepFilterNet repository next to kyone or pass --deepfilter-model", DefaultModel)
}

var (
	libMu     sync.Mutex
	libCached = map[string]*Lib{}
)

// OpenLib loads the library at path (once per path).
func OpenLib(path string) (*Lib, error) {
	libMu.Lock()
	defer libMu.Unlock()
	if l := libCached[path]; l != nil {
		return l, nil
	}
	h, err := openLibrary(path)
	if err != nil {
		return nil, fmt.Errorf("loading %s: %w", path, err)
	}
	l := &Lib{handle: h}
	if err := l.bind(); err != nil {
		return nil, err
	}
	libCached[path] = l
	return l, nil
}

// bind resolves the C functions; purego panics on a missing symbol, which is
// turned into an error here.
func (l *Lib) bind() (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("not a usable DeepFilterNet C library (%v)", r)
		}
	}()
	purego.RegisterLibFunc(&l.create, l.handle, "df_create")
	purego.RegisterLibFunc(&l.free, l.handle, "df_free")
	purego.RegisterLibFunc(&l.frameLength, l.handle, "df_get_frame_length")
	purego.RegisterLibFunc(&l.process, l.handle, "df_process_frame")
	purego.RegisterLibFunc(&l.setAtten, l.handle, "df_set_atten_lim")
	purego.RegisterLibFunc(&l.setPF, l.handle, "df_set_post_filter_beta")
	return nil
}

// Stream is one real-time DeepFilterNet instance (mono, 48 kHz). It processes
// fixed frames of FrameLength() samples. Not safe for concurrent use.
type Stream struct {
	lib   *Lib
	state uintptr
	frame int

	model           string
	attenDB, pfBeta float64
	latencyOnce     sync.Once
	latency         int
}

// NewStream creates a streaming denoiser from a model archive. attenDB is the
// maximum noise reduction in dB; postFilterBeta of 0 disables the post-filter
// (useful values are between 0.02 and 0.05).
func (l *Lib) NewStream(model string, attenDB, postFilterBeta float64) (*Stream, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, fmt.Errorf("DeepFilterNet model: %w", err)
	}
	st := l.create(model, float32(attenDB), 0)
	if st == 0 {
		return nil, errors.New("DeepFilterNet could not load the model (is it a DeepFilterNet ONNX tar.gz?)")
	}
	s := &Stream{lib: l, state: st, frame: int(l.frameLength(st)), model: model, attenDB: attenDB, pfBeta: postFilterBeta}
	if s.frame <= 0 || s.frame > 1<<16 {
		l.free(st)
		return nil, fmt.Errorf("DeepFilterNet reported an invalid frame length %d", s.frame)
	}
	l.setPF(st, float32(postFilterBeta))
	runtime.SetFinalizer(s, func(s *Stream) { s.Close() })
	return s, nil
}

// FrameLength is the number of samples per call to Process (480 = 10 ms).
func (s *Stream) FrameLength() int { return s.frame }

// Latency is the delay, in samples, between a sample entering Process and its
// denoised version leaving it, measured once on a synthetic voice with a
// separate instance of the model (the models differ: 10 ms for the low-latency
// one). It is on top of nothing else: with callbacks of exactly FrameLength()
// samples it is the whole algorithmic delay.
func (s *Stream) Latency() int {
	s.latencyOnce.Do(func() { s.latency = s.measureLatency() })
	return s.latency
}

// measureLatency feeds a voice-like signal through a fresh instance and finds
// the delay that best aligns its output with the noise-free voice.
func (s *Stream) measureLatency() int {
	probe, err := s.lib.NewStream(s.model, s.attenDB, s.pfBeta)
	if err != nil {
		return s.frame // fall back to one frame, the minimum for any model
	}
	defer probe.Close()
	noisy, clean := SyntheticVoice(3, 0.01)
	out := make([]float32, len(noisy))
	for i := 0; i+s.frame <= len(noisy); i += s.frame {
		probe.Process(noisy[i:i+s.frame], out[i:i+s.frame])
	}
	best, bestV := s.frame, math.Inf(-1)
	for lag := 0; lag < 8*s.frame; lag++ {
		var v float64
		for i := SampleRate; i < 2*SampleRate; i += 2 {
			v += float64(out[i]) * float64(clean[i-lag])
		}
		if v > bestV {
			best, bestV = lag, v
		}
	}
	return best
}

// SetAttenuation changes the maximum noise reduction while running.
func (s *Stream) SetAttenuation(db float64) { s.lib.setAtten(s.state, float32(db)) }

// Process denoises one frame: in and out must both hold FrameLength() samples.
// It returns the frame's local SNR in dB.
func (s *Stream) Process(in, out []float32) float32 {
	return s.lib.process(s.state, unsafe.Pointer(&in[0]), unsafe.Pointer(&out[0]))
}

// Close frees the instance; it is safe to call more than once.
func (s *Stream) Close() {
	if s.state != 0 {
		s.lib.free(s.state)
		s.state = 0
	}
}
