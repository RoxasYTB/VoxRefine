package audio

import (
	"math"

	"kyone/internal/dsp"
)

const (
	resampleHalfTaps = 32   // zero crossings on each side of the kernel
	resamplePhases   = 1024 // fractional-delay table resolution
	kaiserBeta       = 9.0  // ≈ 90 dB stop-band attenuation
)

// Resample converts b to the target sample rate with a windowed-sinc
// (Kaiser) interpolator. The input is returned unchanged if the rates match.
func Resample(b *Buffer, rate int) *Buffer {
	if b.Rate == rate || b.Frames() == 0 {
		return b
	}
	ratio := float64(b.Rate) / float64(rate) // input samples per output sample
	cutoff := math.Min(1, 1/ratio) * 0.97    // normalised to input Nyquist
	kernel := buildKernel(cutoff)

	outFrames := int(math.Ceil(float64(b.Frames()) / ratio))
	out := NewBuffer(rate, b.Channels(), outFrames)
	width := resampleHalfTaps
	if cutoff < 1 { // downsampling: the kernel stretches by 1/cutoff
		width = int(math.Ceil(float64(resampleHalfTaps) / cutoff))
	}
	for c, in := range b.Ch {
		dst := out.Ch[c]
		n := len(in)
		for i := range dst {
			pos := float64(i) * ratio
			base := int(math.Floor(pos))
			frac := pos - float64(base)
			var acc float64
			for j := -width + 1; j <= width; j++ {
				idx := base + j
				if idx < 0 || idx >= n {
					continue
				}
				// distance from sample idx to the output position, in input samples
				d := (float64(j) - frac) * cutoff
				acc += float64(in[idx]) * kernel.at(d)
			}
			dst[i] = float32(acc * cutoff)
		}
	}
	return out
}

// kaiserKernel tabulates sinc(d)·kaiser(d/half) for d in [-half, half].
type kaiserKernel struct {
	table []float64
}

func buildKernel(cutoff float64) *kaiserKernel {
	n := resampleHalfTaps * resamplePhases
	k := &kaiserKernel{table: make([]float64, n+2)}
	i0b := dsp.BesselI0(kaiserBeta)
	for i := 0; i <= n; i++ {
		d := float64(i) / resamplePhases // 0..half
		w := dsp.BesselI0(kaiserBeta*math.Sqrt(1-sq(d/resampleHalfTaps))) / i0b
		k.table[i] = sinc(d) * w
	}
	k.table[n+1] = 0
	return k
}

func (k *kaiserKernel) at(d float64) float64 {
	d = math.Abs(d)
	if d >= resampleHalfTaps {
		return 0
	}
	p := d * resamplePhases
	i := int(p)
	f := p - float64(i)
	return k.table[i]*(1-f) + k.table[i+1]*f
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

func sq(x float64) float64 { return x * x }
