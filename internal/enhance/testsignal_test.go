package enhance

import (
	"math"
	"math/rand"

	"kyone/internal/audio"
)

const testRate = 48000

// syntheticSpeech returns a voiced-speech-like signal: a harmonic source with
// formant-shaped spectrum and a 3 Hz syllable envelope, gated into phrases
// separated by pauses. active[i] reports whether sample i is inside a phrase.
func syntheticSpeech(seconds float64, seed int64) (x []float32, active []bool) {
	rng := rand.New(rand.NewSource(seed))
	n := int(seconds * testRate)
	x = make([]float32, n)
	active = make([]bool, n)
	phase := 0.0
	var offset [41]float64 // fixed random phase per harmonic, like a real glottal source
	for h := range offset {
		offset[h] = 2 * math.Pi * rng.Float64()
	}
	for i := range x {
		t := float64(i) / testRate
		// Phrases of 1.6 s separated by 0.9 s pauses.
		inPhrase := math.Mod(t, 2.5) < 1.6
		active[i] = inPhrase
		if !inPhrase {
			continue
		}
		env := 0.5 * (1 + math.Sin(2*math.Pi*3*t))
		f0 := 120 + 25*math.Sin(2*math.Pi*0.7*t)
		phase += 2 * math.Pi * f0 / testRate
		var s float64
		for h := 1; h <= 40; h++ {
			f := f0 * float64(h)
			if f > 7000 {
				break
			}
			// Two formants near 600 Hz and 1800 Hz.
			a := math.Exp(-sq((f-600)/350)) + 0.6*math.Exp(-sq((f-1800)/500)) + 0.05
			s += a * math.Sin(float64(h)*phase+offset[h])
		}
		x[i] = float32(0.12 * env * s / 3)
	}
	return x, active
}

func sq(x float64) float64 { return x * x }

func whiteNoise(n int, sigma float64, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(sigma * rng.NormFloat64())
	}
	return out
}

func mix(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range out {
		out[i] = a[i] + b[i]
	}
	return out
}

func mono(x []float32) *audio.Buffer {
	return &audio.Buffer{Rate: testRate, Ch: [][]float32{append([]float32(nil), x...)}}
}

// rmsDB is the RMS of x[i] over the samples where sel[i] == want, in dBFS.
func rmsDB(x []float32, sel []bool, want bool) float64 {
	var s float64
	var n int
	for i, v := range x {
		if sel[i] == want {
			s += float64(v) * float64(v)
			n++
		}
	}
	if n == 0 || s == 0 {
		return -200
	}
	return 10 * math.Log10(s/float64(n))
}

// errDB is the energy of (x - ref) relative to the energy of ref, in dB.
func errDB(x, ref []float32, sel []bool, want bool) float64 {
	var e, r float64
	for i := range x {
		if sel[i] == want {
			d := float64(x[i] - ref[i])
			e += d * d
			r += float64(ref[i]) * float64(ref[i])
		}
	}
	return 10 * math.Log10(e/r)
}
