// Package dsp holds the low-level signal-processing primitives used by the
// enhancement pipeline: FFT, biquad filters and a few numeric helpers.
package dsp

import (
	"math"
	"math/bits"
)

// FFT is an in-place radix-2 complex FFT of a fixed power-of-two size.
// It is safe for concurrent use: it keeps no scratch state.
type FFT struct {
	n   int
	rev []int
	tw  []complex128 // exp(-2πik/n) for k < n/2
}

// NewFFT returns an FFT of size n, which must be a power of two >= 2.
func NewFFT(n int) *FFT {
	if n < 2 || n&(n-1) != 0 {
		panic("dsp: FFT size must be a power of two >= 2")
	}
	f := &FFT{n: n, rev: make([]int, n), tw: make([]complex128, n/2)}
	shift := bits.UintSize - bits.Len(uint(n-1))
	for i := range f.rev {
		f.rev[i] = int(bits.Reverse(uint(i)) >> shift)
	}
	for k := range f.tw {
		a := -2 * math.Pi * float64(k) / float64(n)
		f.tw[k] = complex(math.Cos(a), math.Sin(a))
	}
	return f
}

// Size returns the transform length.
func (f *FFT) Size() int { return f.n }

// Forward computes the DFT of x in place.
func (f *FFT) Forward(x []complex128) {
	n := f.n
	for i, j := range f.rev {
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		step := n / size
		for i := 0; i < n; i += size {
			for j, k := 0, 0; j < half; j, k = j+1, k+step {
				u := x[i+j]
				v := x[i+j+half] * f.tw[k]
				x[i+j] = u + v
				x[i+j+half] = u - v
			}
		}
	}
}

// Inverse computes the inverse DFT of x in place, including the 1/n scaling.
func (f *FFT) Inverse(x []complex128) {
	for i, v := range x {
		x[i] = complex(real(v), -imag(v))
	}
	f.Forward(x)
	s := 1 / float64(f.n)
	for i, v := range x {
		x[i] = complex(real(v)*s, -imag(v)*s)
	}
}

// RFFT is a real-input FFT of size n built on a complex FFT of size n/2.
// Unlike FFT it owns scratch memory, so one instance must not be shared
// between goroutines.
type RFFT struct {
	n    int
	half *FFT
	w    []complex128 // exp(-2πik/n), k = 0..n/2
	z    []complex128
}

// NewRFFT returns a real FFT of size n (power of two >= 4).
func NewRFFT(n int) *RFFT {
	if n < 4 {
		panic("dsp: RFFT size must be >= 4")
	}
	r := &RFFT{n: n, half: NewFFT(n / 2), w: make([]complex128, n/2+1), z: make([]complex128, n/2)}
	for k := range r.w {
		a := -2 * math.Pi * float64(k) / float64(n)
		r.w[k] = complex(math.Cos(a), math.Sin(a))
	}
	return r
}

// Size returns the number of real samples per transform.
func (r *RFFT) Size() int { return r.n }

// Bins returns the number of spectrum bins, n/2+1.
func (r *RFFT) Bins() int { return r.n/2 + 1 }

// Forward transforms n real samples into n/2+1 spectrum bins.
func (r *RFFT) Forward(x []float64, out []complex128) {
	h := r.n / 2
	for m := 0; m < h; m++ {
		r.z[m] = complex(x[2*m], x[2*m+1])
	}
	r.half.Forward(r.z)
	for k := 0; k <= h; k++ {
		zk := r.z[k%h]
		zc := r.z[(h-k)%h]
		zc = complex(real(zc), -imag(zc))
		xe := (zk + zc) * 0.5
		xo := (zk - zc) * complex(0, -0.5)
		out[k] = xe + r.w[k]*xo
	}
}

// Inverse transforms n/2+1 spectrum bins back into n real samples.
// Inverse(Forward(x)) reproduces x.
func (r *RFFT) Inverse(spec []complex128, out []float64) {
	h := r.n / 2
	for k := 0; k < h; k++ {
		xk := spec[k]
		xc := spec[h-k]
		xc = complex(real(xc), -imag(xc))
		xe := (xk + xc) * 0.5
		w := r.w[k]
		xo := (xk - xc) * 0.5 * complex(real(w), -imag(w))
		r.z[k] = xe + complex(0, 1)*xo
	}
	r.half.Inverse(r.z)
	for m := 0; m < h; m++ {
		out[2*m] = real(r.z[m])
		out[2*m+1] = imag(r.z[m])
	}
}
