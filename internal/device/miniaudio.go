//go:build cgo

package device

import (
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/gen2brain/malgo"
)

// withContext runs fn with an initialised miniaudio context.
func withContext(fn func(ctx *malgo.AllocatedContext) error) error {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return fmt.Errorf("audio backend: %w", err)
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()
	return fn(ctx)
}

func toInfos(devs []malgo.DeviceInfo) []Info {
	out := make([]Info, len(devs))
	for i := range devs {
		out[i] = Info{Name: devs[i].Name(), Default: devs[i].IsDefault != 0}
	}
	return out
}

// List returns the capture or playback devices the system exposes.
func List(kind Kind) ([]Info, error) {
	var infos []Info
	err := withContext(func(ctx *malgo.AllocatedContext) error {
		t := malgo.Capture
		if kind == Playback {
			t = malgo.Playback
		}
		devs, err := ctx.Devices(t)
		if err != nil {
			return err
		}
		infos = toInfos(devs)
		return nil
	})
	return infos, err
}

// Session is a running capture → process → playback pipeline.
type Session struct {
	ctx         *malgo.AllocatedContext
	capture     *malgo.Device
	playback    *malgo.Device
	desc        string
	underruns   atomic.Uint64
	overruns    atomic.Uint64
	frames      atomic.Uint64
	inPeak      peakMeter
	outPeak     peakMeter
	lastInPeak  atomic.Uint32
	lastOutPeak atomic.Uint32
	stopped     atomic.Bool
}

// Start opens the devices and begins processing. process receives one block of
// mono microphone samples and must fill out (same length) with the result; it
// runs on the audio thread, so it must be fast and must not block.
func Start(cfg Config, process func(in, out []float32)) (*Session, error) {
	block := cfg.BlockMS
	if block <= 0 {
		block = 10
	}
	frames := SampleRate * block / 1000

	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("audio backend: %w", err)
	}
	s := &Session{ctx: ctx}
	fail := func(err error) (*Session, error) {
		s.Stop()
		return nil, err
	}

	capDevs, err := ctx.Devices(malgo.Capture)
	if err != nil {
		return fail(err)
	}
	playDevs, err := ctx.Devices(malgo.Playback)
	if err != nil {
		return fail(err)
	}
	ci, err := match(toInfos(capDevs), cfg.Input, Capture)
	if err != nil {
		return fail(err)
	}
	pi, err := match(toInfos(playDevs), cfg.Output, Playback)
	if err != nil {
		return fail(err)
	}

	// Samples travel capture → process → ring → playback. The ring absorbs the
	// small jitter between the two device clocks; it is trimmed if one runs
	// ahead, and playback is held back until a little audio has accumulated.
	r := newRing(SampleRate) // one second of headroom
	prefill := 2 * frames
	maxFill := 6 * frames
	var primed atomic.Bool

	scratchOut := make([]float32, 8*frames)
	capCfg := malgo.DefaultDeviceConfig(malgo.Capture)
	capCfg.SampleRate = SampleRate
	capCfg.PeriodSizeInFrames = uint32(frames)
	capCfg.Periods = 2
	capCfg.Capture.Format = malgo.FormatF32
	capCfg.Capture.Channels = 1
	if ci >= 0 {
		capCfg.Capture.DeviceID = capDevs[ci].ID.Pointer()
	}
	capDev, err := malgo.InitDevice(ctx.Context, capCfg, malgo.DeviceCallbacks{
		Data: func(_, in []byte, n uint32) {
			if len(in) == 0 {
				return
			}
			samples := unsafe.Slice((*float32)(unsafe.Pointer(&in[0])), int(n))
			if int(n) > len(scratchOut) {
				scratchOut = make([]float32, int(n))
			}
			out := scratchOut[:n]
			s.inPeak.observe(samples)
			process(samples, out)
			s.outPeak.observe(out)
			s.frames.Add(uint64(n))
			if d := r.push(out); d > 0 {
				s.overruns.Add(1)
			}
			if r.len() > maxFill { // the playback clock is slower: stay close
				r.dropOldest(prefill)
				s.overruns.Add(1)
			}
		},
	})
	if err != nil {
		return fail(fmt.Errorf("opening the microphone: %w", err))
	}
	s.capture = capDev

	playCfg := malgo.DefaultDeviceConfig(malgo.Playback)
	playCfg.SampleRate = SampleRate
	playCfg.PeriodSizeInFrames = uint32(frames)
	playCfg.Periods = 2
	playCfg.Playback.Format = malgo.FormatF32
	playCfg.Playback.Channels = 2
	if pi >= 0 {
		playCfg.Playback.DeviceID = playDevs[pi].ID.Pointer()
	}
	mono := make([]float32, 8*frames)
	playDev, err := malgo.InitDevice(ctx.Context, playCfg, malgo.DeviceCallbacks{
		Data: func(out, _ []byte, n uint32) {
			if len(out) == 0 {
				return
			}
			dst := unsafe.Slice((*float32)(unsafe.Pointer(&out[0])), int(n)*2)
			if int(n) > len(mono) {
				mono = make([]float32, int(n))
			}
			src := mono[:n]
			if !primed.Load() {
				if r.len() < prefill {
					clear(dst)
					return
				}
				primed.Store(true)
			}
			got := r.pop(src)
			if got < int(n) {
				clear(src[got:])
				s.underruns.Add(1)
				primed.Store(false) // re-buffer instead of stuttering
			}
			for i, v := range src {
				dst[2*i], dst[2*i+1] = v, v
			}
		},
	})
	if err != nil {
		return fail(fmt.Errorf("opening the output device: %w", err))
	}
	s.playback = playDev

	capName, playName := "default microphone", "default output"
	if ci >= 0 {
		capName = capDevs[ci].Name()
	}
	if pi >= 0 {
		playName = playDevs[pi].Name()
	}
	s.desc = fmt.Sprintf("%s → %s", capName, playName)

	if err := playDev.Start(); err != nil {
		return fail(fmt.Errorf("starting the output: %w", err))
	}
	if err := capDev.Start(); err != nil {
		return fail(fmt.Errorf("starting the microphone: %w", err))
	}
	return s, nil
}

// Description names the devices in use.
func (s *Session) Description() string { return s.desc }

// Stop closes the devices. It is safe to call more than once.
func (s *Session) Stop() {
	if s == nil || !s.stopped.CompareAndSwap(false, true) {
		return
	}
	if s.capture != nil {
		s.capture.Uninit()
	}
	if s.playback != nil {
		s.playback.Uninit()
	}
	if s.ctx != nil {
		_ = s.ctx.Uninit()
		s.ctx.Free()
	}
}

// Stats returns the counters; the peaks cover the time since the last call.
func (s *Session) Stats() Stats {
	return Stats{
		Underruns: s.underruns.Load(), Overruns: s.overruns.Load(), Frames: s.frames.Load(),
		InputPeak: s.inPeak.take(), OutputPeak: s.outPeak.take(),
	}
}
