package dsp

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

func TestFFTMatchesNaiveDFT(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewSource(1))
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(rng.NormFloat64(), rng.NormFloat64())
	}
	want := make([]complex128, n)
	for k := range want {
		for i, v := range x {
			want[k] += v * cmplx.Exp(complex(0, -2*math.Pi*float64(k*i)/n))
		}
	}
	got := append([]complex128(nil), x...)
	NewFFT(n).Forward(got)
	for k := range got {
		if cmplx.Abs(got[k]-want[k]) > 1e-9 {
			t.Fatalf("bin %d: got %v want %v", k, got[k], want[k])
		}
	}
	NewFFT(n).Inverse(got)
	for i := range got {
		if cmplx.Abs(got[i]-x[i]) > 1e-9 {
			t.Fatalf("inverse sample %d: got %v want %v", i, got[i], x[i])
		}
	}
}

func TestRFFTMatchesComplexFFT(t *testing.T) {
	const n = 256
	rng := rand.New(rand.NewSource(2))
	x := make([]float64, n)
	c := make([]complex128, n)
	for i := range x {
		x[i] = rng.NormFloat64()
		c[i] = complex(x[i], 0)
	}
	NewFFT(n).Forward(c)
	r := NewRFFT(n)
	spec := make([]complex128, r.Bins())
	r.Forward(x, spec)
	for k := range spec {
		if cmplx.Abs(spec[k]-c[k]) > 1e-9 {
			t.Fatalf("bin %d: got %v want %v", k, spec[k], c[k])
		}
	}
	back := make([]float64, n)
	r.Inverse(spec, back)
	for i := range back {
		if math.Abs(back[i]-x[i]) > 1e-9 {
			t.Fatalf("sample %d: got %v want %v", i, back[i], x[i])
		}
	}
}

func TestBiquadResponses(t *testing.T) {
	const rate = 48000
	tests := []struct {
		name string
		c    Cascade
		freq float64
		want float64 // dB
		tol  float64
	}{
		{"butterworth hp cutoff", ButterworthHighPass(rate, 100, 4), 100, -3.01, 0.1},
		{"butterworth hp passband", ButterworthHighPass(rate, 100, 4), 5000, 0, 0.05},
		{"butterworth hp stopband", ButterworthHighPass(rate, 100, 4), 25, -48, 3},
		{"peaking centre", Cascade{Peaking(rate, 3000, 1, 6)}, 3000, 6, 0.05},
		{"peaking far", Cascade{Peaking(rate, 3000, 1, 6)}, 100, 0, 0.3},
		{"low shelf low", Cascade{LowShelf(rate, 150, 4)}, 20, 4, 0.2},
		{"low shelf high", Cascade{LowShelf(rate, 150, 4)}, 8000, 0, 0.1},
		{"high shelf high", Cascade{HighShelf(rate, 4000, -5)}, 20000, -5, 0.3},
		{"high shelf low", Cascade{HighShelf(rate, 4000, -5)}, 200, 0, 0.1},
		{"notch", Cascade{Notch(rate, 50, 10)}, 50, -100, 100},
	}
	for _, tc := range tests {
		got := DB(cmplx.Abs(tc.c.Response(tc.freq, rate)))
		if tc.name == "notch" {
			if got > -40 {
				t.Errorf("%s: got %.1f dB, want deep notch", tc.name, got)
			}
			continue
		}
		if math.Abs(got-tc.want) > tc.tol {
			t.Errorf("%s: got %.2f dB want %.2f±%.2f", tc.name, got, tc.want, tc.tol)
		}
	}
}

func TestBiquadProcessMatchesResponse(t *testing.T) {
	const rate = 48000
	c := Cascade{Peaking(rate, 2000, 0.8, 9)}
	const f = 2000.0
	var peak float64
	for i := 0; i < rate; i++ {
		y := c.Process(math.Sin(2 * math.Pi * f * float64(i) / rate))
		if i > rate/2 && math.Abs(y) > peak {
			peak = math.Abs(y)
		}
	}
	if got := DB(peak); math.Abs(got-9) > 0.1 {
		t.Fatalf("steady-state gain %.2f dB, want 9", got)
	}
}
