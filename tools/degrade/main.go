// Command degrade makes a clean voice recording worse in controlled ways
// (background noise, mains hum, room reverb) so that kyone can be evaluated
// against a known reference. It is a development helper, not part of kyone.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"

	"kyone/internal/audio"
	"kyone/internal/dsp"
	"kyone/internal/loudness"
)

func main() {
	snr := flag.Float64("snr", 15, "signal-to-noise ratio of the added broadband noise in dB (speech level vs noise level)")
	hum := flag.Float64("hum", -200, "mains hum level in dBFS (50 Hz + harmonics); -200 = none")
	rt60 := flag.Float64("rt60", 0, "room reverberation time in seconds; 0 = dry")
	wet := flag.Float64("wet", 0.5, "reverb tail level relative to the direct sound (linear energy ratio)")
	clicks := flag.Float64("clicks", -200, "keyboard-like clicks (about 4 per second) at this level in dB relative to the speech level; -200 = none")
	seed := flag.Int64("seed", 1, "random seed")
	flag.Parse()
	if flag.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "usage: degrade [options] in.wav out.wav")
		os.Exit(2)
	}
	if err := run(flag.Arg(0), flag.Arg(1), *snr, *hum, *rt60, *wet, *clicks, *seed); err != nil {
		fmt.Fprintln(os.Stderr, "degrade:", err)
		os.Exit(1)
	}
}

func run(inPath, outPath string, snr, humDB, rt60, wet, clicksDB float64, seed int64) error {
	f, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := audio.ReadWAV(f)
	if err != nil {
		return err
	}
	b = b.ToChannels(1)
	rng := rand.New(rand.NewSource(seed))
	x := b.Ch[0]

	if rt60 > 0 {
		x = reverberate(x, b.Rate, rt60, wet, rng)
	}

	_, speechDB := loudness.Levels(&audio.Buffer{Rate: b.Rate, Ch: [][]float32{x}})
	noiseRMS := math.Pow(10, (speechDB-snr)/20)
	// Pink-ish noise: white plus two low-pass-filtered copies, so the
	// spectrum falls with frequency like a fan or room tone.
	lp1 := dsp.LowPass(float64(b.Rate), 600, 0.7)
	lp2 := dsp.LowPass(float64(b.Rate), 3000, 0.7)
	noise := make([]float64, len(x))
	var ss float64
	for i := range noise {
		w := rng.NormFloat64()
		noise[i] = 0.4*w + 1.2*lp1.Process(w) + 0.8*lp2.Process(w)
		ss += noise[i] * noise[i]
	}
	scale := noiseRMS / math.Sqrt(ss/float64(len(noise)))
	humAmp := math.Pow(10, humDB/20)
	out := audio.NewBuffer(b.Rate, 1, len(x))
	for i := range out.Ch[0] {
		v := float64(x[i]) + scale*noise[i]
		if humDB > -150 {
			ph := 2 * math.Pi * float64(i) / float64(b.Rate)
			v += humAmp * (math.Sin(ph*50) + 0.5*math.Sin(ph*100) + 0.25*math.Sin(ph*150))
		}
		out.Ch[0][i] = float32(v)
	}

	if clicksDB > -150 {
		addClicks(out.Ch[0], b.Rate, math.Pow(10, (speechDB+clicksDB)/20), rng)
	}

	o, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer o.Close()
	return audio.WriteWAV(o, out, 16)
}

// reverberate convolves x with a synthetic room impulse response (decaying
// Gaussian noise) by FFT, keeping the original length.
func reverberate(x []float32, rate int, rt60, wet float64, rng *rand.Rand) []float32 {
	irLen := int(rt60 * 1.2 * float64(rate))
	ir := make([]float64, irLen)
	var tail float64
	start := int(0.012 * float64(rate)) // tail begins after ~12 ms
	for i := start; i < irLen; i++ {
		env := math.Pow(10, -3*float64(i)/(rt60*float64(rate)))
		ir[i] = rng.NormFloat64() * env
		tail += ir[i] * ir[i]
	}
	k := math.Sqrt(wet / tail) // direct tap is 1, so wet is the tail/direct energy ratio
	for i := start; i < irLen; i++ {
		ir[i] *= k
	}
	ir[0] = 1

	n := 1
	for n < len(x)+irLen {
		n <<= 1
	}
	fft := dsp.NewFFT(n)
	a := make([]complex128, n)
	h := make([]complex128, n)
	for i, v := range x {
		a[i] = complex(float64(v), 0)
	}
	for i, v := range ir {
		h[i] = complex(v, 0)
	}
	fft.Forward(a)
	fft.Forward(h)
	for i := range a {
		a[i] *= h[i]
	}
	fft.Inverse(a)
	out := make([]float32, len(x))
	for i := range out {
		out[i] = float32(real(a[i]))
	}
	return out
}

// addClicks overlays keyboard-like transients: short decaying bursts of
// band-limited noise (1.5–7 kHz) at random times, about four a second.
func addClicks(x []float32, rate int, amp float64, rng *rand.Rand) {
	bp := dsp.Cascade{dsp.HighPass(float64(rate), 1500, 0.7), dsp.LowPass(float64(rate), 7000, 0.7)}
	burst := int(0.012 * float64(rate))
	for t := 0.1; t < float64(len(x))/float64(rate)-0.05; t += 0.1 + rng.ExpFloat64()*0.15 {
		start := int(t * float64(rate))
		for i := 0; i < burst && start+i < len(x); i++ {
			env := math.Exp(-float64(i) / (0.003 * float64(rate)))
			x[start+i] += float32(amp * env * bp.Process(rng.NormFloat64()))
		}
	}
}
