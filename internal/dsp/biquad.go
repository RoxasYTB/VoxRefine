package dsp

import (
	"math"
	"math/cmplx"
)

// Biquad is a second-order IIR section in transposed direct form II.
// Coefficients are normalised so that a0 == 1.
type Biquad struct {
	b0, b1, b2, a1, a2 float64
	z1, z2             float64
}

// NewBiquad builds a section from raw (unnormalised) coefficients.
func NewBiquad(b0, b1, b2, a0, a1, a2 float64) *Biquad {
	return &Biquad{b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0}
}

// Process filters one sample.
func (b *Biquad) Process(x float64) float64 {
	y := b.b0*x + b.z1
	b.z1 = b.b1*x - b.a1*y + b.z2
	b.z2 = b.b2*x - b.a2*y
	return y
}

// Reset clears the filter state.
func (b *Biquad) Reset() { b.z1, b.z2 = 0, 0 }

// Clone returns a copy of the section with cleared state.
func (b *Biquad) Clone() *Biquad {
	c := *b
	c.Reset()
	return &c
}

// Response returns the complex frequency response at freq Hz.
func (b *Biquad) Response(freq, rate float64) complex128 {
	w := 2 * math.Pi * freq / rate
	z1 := cmplx.Exp(complex(0, -w))
	z2 := z1 * z1
	num := complex(b.b0, 0) + complex(b.b1, 0)*z1 + complex(b.b2, 0)*z2
	den := complex(1, 0) + complex(b.a1, 0)*z1 + complex(b.a2, 0)*z2
	return num / den
}

// Cascade is a series of biquad sections.
type Cascade []*Biquad

// Process filters one sample through every section.
func (c Cascade) Process(x float64) float64 {
	for _, s := range c {
		x = s.Process(x)
	}
	return x
}

// Reset clears the state of every section.
func (c Cascade) Reset() {
	for _, s := range c {
		s.Reset()
	}
}

// Clone returns a copy of the cascade with cleared state.
func (c Cascade) Clone() Cascade {
	out := make(Cascade, len(c))
	for i, s := range c {
		out[i] = s.Clone()
	}
	return out
}

// Response returns the combined complex response at freq Hz.
func (c Cascade) Response(freq, rate float64) complex128 {
	h := complex(1, 0)
	for _, s := range c {
		h *= s.Response(freq, rate)
	}
	return h
}

// ProcessFloat32 filters buf in place.
func (c Cascade) ProcessFloat32(buf []float32) {
	for i, v := range buf {
		buf[i] = float32(c.Process(float64(v)))
	}
}

// The constructors below follow the RBJ audio-EQ cookbook.

// HighPass returns a 2nd-order high-pass at fc with quality factor q.
func HighPass(rate, fc, q float64) *Biquad {
	w := 2 * math.Pi * fc / rate
	cw, sw := math.Cos(w), math.Sin(w)
	alpha := sw / (2 * q)
	return NewBiquad((1+cw)/2, -(1 + cw), (1+cw)/2, 1+alpha, -2*cw, 1-alpha)
}

// LowPass returns a 2nd-order low-pass at fc with quality factor q.
func LowPass(rate, fc, q float64) *Biquad {
	w := 2 * math.Pi * fc / rate
	cw, sw := math.Cos(w), math.Sin(w)
	alpha := sw / (2 * q)
	return NewBiquad((1-cw)/2, 1-cw, (1-cw)/2, 1+alpha, -2*cw, 1-alpha)
}

// Peaking returns a peaking EQ section with the given gain in dB.
func Peaking(rate, fc, q, gainDB float64) *Biquad {
	a := math.Pow(10, gainDB/40)
	w := 2 * math.Pi * fc / rate
	cw, sw := math.Cos(w), math.Sin(w)
	alpha := sw / (2 * q)
	return NewBiquad(1+alpha*a, -2*cw, 1-alpha*a, 1+alpha/a, -2*cw, 1-alpha/a)
}

// LowShelf returns a low-shelf section (slope 1) with the given gain in dB.
func LowShelf(rate, fc, gainDB float64) *Biquad {
	a := math.Pow(10, gainDB/40)
	w := 2 * math.Pi * fc / rate
	cw, sw := math.Cos(w), math.Sin(w)
	alpha := sw / 2 * math.Sqrt(2)
	t := 2 * math.Sqrt(a) * alpha
	return NewBiquad(
		a*((a+1)-(a-1)*cw+t),
		2*a*((a-1)-(a+1)*cw),
		a*((a+1)-(a-1)*cw-t),
		(a+1)+(a-1)*cw+t,
		-2*((a-1)+(a+1)*cw),
		(a+1)+(a-1)*cw-t,
	)
}

// HighShelf returns a high-shelf section (slope 1) with the given gain in dB.
func HighShelf(rate, fc, gainDB float64) *Biquad {
	a := math.Pow(10, gainDB/40)
	w := 2 * math.Pi * fc / rate
	cw, sw := math.Cos(w), math.Sin(w)
	alpha := sw / 2 * math.Sqrt(2)
	t := 2 * math.Sqrt(a) * alpha
	return NewBiquad(
		a*((a+1)+(a-1)*cw+t),
		-2*a*((a-1)+(a+1)*cw),
		a*((a+1)+(a-1)*cw-t),
		(a+1)-(a-1)*cw+t,
		2*((a-1)-(a+1)*cw),
		(a+1)-(a-1)*cw-t,
	)
}

// Notch returns a notch filter at fc with quality factor q.
func Notch(rate, fc, q float64) *Biquad {
	w := 2 * math.Pi * fc / rate
	cw, sw := math.Cos(w), math.Sin(w)
	alpha := sw / (2 * q)
	return NewBiquad(1, -2*cw, 1, 1+alpha, -2*cw, 1-alpha)
}

// ButterworthHighPass returns an even-order Butterworth high-pass as a
// cascade of order/2 biquads.
func ButterworthHighPass(rate, fc float64, order int) Cascade {
	if order < 2 || order%2 != 0 {
		panic("dsp: Butterworth order must be an even number >= 2")
	}
	c := make(Cascade, 0, order/2)
	for k := 0; k < order/2; k++ {
		q := 1 / (2 * math.Sin(math.Pi*float64(2*k+1)/float64(2*order)))
		c = append(c, HighPass(rate, fc, q))
	}
	return c
}

// DB converts a linear amplitude to decibels, flooring at -200 dB.
func DB(x float64) float64 {
	if x < 1e-10 {
		return -200
	}
	return 20 * math.Log10(x)
}

// FromDB converts decibels to a linear amplitude.
func FromDB(db float64) float64 { return math.Pow(10, db/20) }

// BesselI0 evaluates the zeroth-order modified Bessel function of the first
// kind, used by Kaiser windows.
func BesselI0(x float64) float64 {
	sum, term := 1.0, 1.0
	for k := 1; k < 60; k++ {
		h := x / (2 * float64(k))
		term *= h * h
		sum += term
		if term < 1e-12*sum {
			break
		}
	}
	return sum
}

// Four-times-oversampled true-peak interpolation: 12 Kaiser-windowed sinc taps
// per phase, evaluated at offsets 1/4, 2/4 and 3/4 between samples.
const tpTaps = 12

var tpKernel = func() (k [3][tpTaps]float64) {
	const beta = 8.0
	i0 := BesselI0(beta)
	for p := 1; p <= 3; p++ {
		for t := 0; t < tpTaps; t++ {
			j := t - tpTaps/2 + 1 // tap offsets -5..6 relative to the left sample
			d := float64(p)/4 - float64(j)
			w := BesselI0(beta*math.Sqrt(math.Max(0, 1-(d/6)*(d/6)))) / i0
			s := 1.0
			if d != 0 {
				s = math.Sin(math.Pi*d) / (math.Pi * d)
			}
			k[p-1][t] = s * w
		}
	}
	return
}()

// TruePeakEnvelope returns, for every sample m, the largest magnitude found
// among x[m] and the three interpolated points between x[m] and x[m+1]
// (4x oversampling, as in ITU-R BS.1770 true-peak metering).
func TruePeakEnvelope(x []float32) []float32 {
	env := make([]float32, len(x))
	TruePeakEnvelopeInto(env, x)
	return env
}

// TruePeakMargin is the number of samples on each side of a sample that the
// interpolator reads. To evaluate a stretch of a longer signal exactly, pass
// the stretch extended by this margin and discard the margin's outputs.
const TruePeakMargin = tpTaps/2 + 1

// TruePeakEnvelopeInto is TruePeakEnvelope writing into dst, which must be as
// long as x.
func TruePeakEnvelopeInto(dst, x []float32) {
	n := len(x)
	for m := 0; m < n; m++ {
		peak := math.Abs(float64(x[m]))
		for p := 0; p < 3; p++ {
			var acc float64
			for t := 0; t < tpTaps; t++ {
				i := m + t - tpTaps/2 + 1
				if i >= 0 && i < n {
					acc += float64(x[i]) * tpKernel[p][t]
				}
			}
			if a := math.Abs(acc); a > peak {
				peak = a
			}
		}
		dst[m] = float32(peak)
	}
}

// TruePeak returns the largest true (4x oversampled) peak magnitude in x,
// working in blocks so that no signal-sized buffer is allocated.
func TruePeak(x []float32) float64 {
	const block = 1 << 15
	env := make([]float32, block+2*TruePeakMargin)
	var peak float32
	for a := 0; a < len(x); a += block {
		b := min(a+block, len(x))
		lo, hi := max(a-TruePeakMargin, 0), min(b+TruePeakMargin, len(x))
		TruePeakEnvelopeInto(env[:hi-lo], x[lo:hi])
		for _, v := range env[a-lo : b-lo] {
			if v > peak {
				peak = v
			}
		}
	}
	return float64(peak)
}

// BandPass returns a 2nd-order band-pass at fc with quality factor q and
// 0 dB gain at the centre (RBJ "constant peak gain" form).
func BandPass(rate, fc, q float64) *Biquad {
	w := 2 * math.Pi * fc / rate
	cw, sw := math.Cos(w), math.Sin(w)
	alpha := sw / (2 * q)
	return NewBiquad(alpha, 0, -alpha, 1+alpha, -2*cw, 1-alpha)
}
