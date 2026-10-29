package deepfilter

import (
	"math"
	"math/rand"
)

// SyntheticVoice returns a harmonic, formant-shaped "voice" with 3 Hz syllables
// in phrases of 1.6 s, once over white noise of the given level and once alone.
// It is used to measure delays and noise reduction.
func SyntheticVoice(seconds, noiseSigma float64) (noisy, clean []float32) {
	rng := rand.New(rand.NewSource(1))
	n := int(seconds * SampleRate)
	noisy, clean = make([]float32, n), make([]float32, n)
	phase := 0.0
	for i := range noisy {
		t := float64(i) / SampleRate
		var v float64
		if math.Mod(t, 2.5) < 1.6 {
			env := 0.5 * (1 + math.Sin(2*math.Pi*3*t))
			f0 := 120 + 25*math.Sin(2*math.Pi*0.7*t)
			phase += 2 * math.Pi * f0 / SampleRate
			for h := 1; h <= 30; h++ {
				f := f0 * float64(h)
				if f > 6000 {
					break
				}
				v += (math.Exp(-math.Pow((f-600)/350, 2)) + 0.6*math.Exp(-math.Pow((f-1800)/500, 2)) + 0.05) * math.Sin(float64(h)*phase)
			}
			v *= 0.04 * env
		}
		clean[i] = float32(v)
		noisy[i] = float32(v + noiseSigma*rng.NormFloat64())
	}
	return noisy, clean
}
