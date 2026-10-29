package enhance

import (
	"fmt"
	"math"
	"math/cmplx"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// Neural is an external neural noise suppressor (DeepFilterNet) that replaces
// kyone's own spectral denoiser. The rest of the chain — tone matching,
// de-esser, clarity, compressor, levelling, limiter — is unchanged. It is
// defined here as a pair of functions so that this package does not depend on
// how the model is loaded.
type Neural struct {
	Name string
	// File denoises a whole mono 48 kHz recording, returning the same length,
	// aligned with the input. attenDB (0–100) is the maximum reduction. May be
	// nil if offline processing is not supported.
	File func(x []float32, attenDB float64) ([]float32, error)
	// Open starts a streaming denoiser for the live mode. May be nil.
	Open func(attenDB float64) (StreamDenoiser, error)
}

// StreamDenoiser is a real-time mono 48 kHz denoiser working on fixed frames.
type StreamDenoiser interface {
	// FrameLength is the number of samples per Process call.
	FrameLength() int
	// Latency is the delay in samples between a sample entering Process and
	// its denoised version leaving it.
	Latency() int
	// Process denoises one frame (len(in) == len(out) == FrameLength()).
	Process(in, out []float32)
	// Close releases the denoiser.
	Close()
}

// neuralDenoise runs the external denoiser over every channel of b.
func neuralDenoise(b *audio.Buffer, n *Neural, attenDB float64) error {
	if n.File == nil {
		return fmt.Errorf("%s cannot process files", n.Name)
	}
	for c, ch := range b.Ch {
		y, err := n.File(ch, attenDB)
		if err != nil {
			return fmt.Errorf("%s (channel %d): %w", n.Name, c+1, err)
		}
		if len(y) != len(ch) {
			return fmt.Errorf("%s returned %d samples for %d", n.Name, len(y), len(ch))
		}
		copy(ch, y)
	}
	return nil
}

// ---- tone matching as a filter (for the live neural path) --------------------------
//
// Native live processing applies the tone curve inside its STFT. With the
// neural engine there is no STFT in the signal path, so the curve is turned
// into a minimum-phase FIR (no added delay) that is crossfaded whenever the
// curve is re-learned.

const toneFIRTaps = 256

// minPhaseFIR designs a causal minimum-phase FIR whose magnitude follows gains
// (linear, nBins values on the 1024-point grid), via the cepstral method.
func minPhaseFIR(gains []float64, taps int) []float64 {
	n := (len(gains) - 1) * 2
	fft := dsp.NewFFT(n)
	c := make([]complex128, n)
	for k, g := range gains {
		v := math.Log(math.Max(g, 1e-6))
		c[k] = complex(v, 0)
		if k > 0 && k < n/2 {
			c[n-k] = complex(v, 0)
		}
	}
	fft.Inverse(c) // real cepstrum of the log magnitude
	for i := 1; i < n/2; i++ {
		c[i] = complex(2*real(c[i]), 0) // fold onto the causal side
	}
	c[n/2] = complex(real(c[n/2]), 0)
	for i := n/2 + 1; i < n; i++ {
		c[i] = 0
	}
	fft.Forward(c)
	for k := range c {
		c[k] = cmplx.Exp(c[k])
	}
	fft.Inverse(c)
	h := make([]float64, taps)
	for i := range h {
		h[i] = real(c[i])
		if tail := taps * 3 / 4; i >= tail { // fade the last quarter out
			h[i] *= 0.5 * (1 + math.Cos(math.Pi*float64(i-tail)/float64(taps-tail)))
		}
	}
	return h
}

// adaptiveFIR is an FIR filter whose coefficients can be replaced at any time:
// the output crossfades from the old to the new filter over 50 ms.
type adaptiveFIR struct {
	hist     []float64 // input history, doubled so reads are contiguous
	pos      int
	cur      []float64
	next     []float64
	fade     int
	fadeLen  int
	identity bool
}

func newAdaptiveFIR(taps int, rate float64) *adaptiveFIR {
	f := &adaptiveFIR{hist: make([]float64, 2*taps), cur: make([]float64, taps), fadeLen: int(0.05 * rate), identity: true}
	f.cur[0] = 1
	return f
}

// set starts a crossfade to the filter h (len == taps).
func (f *adaptiveFIR) set(h []float64) {
	if f.next == nil {
		f.next = make([]float64, len(h))
	}
	copy(f.next, h)
	f.fade = f.fadeLen
	f.identity = false
}

func (f *adaptiveFIR) dot(h []float64) float64 {
	taps := len(h)
	// hist[pos+taps-1] is the newest sample; h[0] multiplies it.
	win := f.hist[f.pos : f.pos+taps]
	var s float64
	for i, c := range h {
		s += c * win[taps-1-i]
	}
	return s
}

func (f *adaptiveFIR) process(x float64) float64 {
	taps := len(f.cur)
	// Slide the doubled history: write at pos and pos+taps so that the window
	// hist[pos : pos+taps] is always the last `taps` samples in order.
	f.hist[f.pos] = x
	f.hist[f.pos+taps] = x
	f.pos = (f.pos + 1) % taps
	if f.identity {
		return x
	}
	y := f.dot(f.cur)
	if f.fade > 0 {
		a := 1 - float64(f.fade)/float64(f.fadeLen)
		y = (1-a)*y + a*f.dot(f.next)
		if f.fade--; f.fade == 0 {
			f.cur, f.next = f.next, f.cur
		}
	}
	return y
}
