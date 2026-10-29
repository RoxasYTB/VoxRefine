package enhance

import (
	"math"
	"math/rand"
	"testing"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

func unity(pos int, pow, noise []float64) []float64 {
	g := make([]float64, nBins)
	for k := range g {
		g[k] = 1
	}
	return g
}

func TestResynthesisIsTransparent(t *testing.T) {
	speech, _ := syntheticSpeech(3, 1)
	in := mono(mix(speech, whiteNoise(len(speech), 0.01, 2)))
	stereo := &audio.Buffer{Rate: testRate, Ch: [][]float32{in.Ch[0], append([]float32(nil), in.Ch[0]...)}}
	for i := range stereo.Ch[1] {
		stereo.Ch[1][i] *= 0.5
	}
	want := stereo.Clone()
	process(stereo, nil, unity)
	for c := range want.Ch {
		for i := range want.Ch[c] {
			if d := math.Abs(float64(stereo.Ch[c][i] - want.Ch[c][i])); d > 1e-5 {
				t.Fatalf("ch%d[%d]: diff %g", c, i, d)
			}
		}
	}
}

func TestExpint1(t *testing.T) {
	// Reference values of E1(x) from Abramowitz & Stegun table 5.1.
	for _, tc := range []struct{ x, want float64 }{
		{0.01, 4.037929576538114}, {0.1, 1.822923958419390}, {0.5, 0.559773594776161},
		{1, 0.219383934395520}, {2, 0.048900510708061}, {5, 0.001148295591275},
	} {
		if got := expint1(tc.x); math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("E1(%g) = %.12f, want %.12f", tc.x, got, tc.want)
		}
	}
}

func TestSpectralInactiveWhenDisabled(t *testing.T) {
	speech, _ := syntheticSpeech(1, 1)
	b := mono(speech)
	want := b.Clone()
	Spectral(b, SpectralConfig{})
	for i := range want.Ch[0] {
		if b.Ch[0][i] != want.Ch[0][i] {
			t.Fatal("signal changed with a disabled stage")
		}
	}
}

func TestDenoiseReducesNoiseAndKeepsSpeech(t *testing.T) {
	speech, active := syntheticSpeech(20, 1)
	noise := whiteNoise(len(speech), 0.01, 2) // ≈ -40 dBFS
	noisy := mix(speech, noise)

	b := mono(noisy)
	Spectral(b, SpectralConfig{AttenuationDB: 25, Overshoot: 1.2})
	out := b.Ch[0]

	// Noise in the pauses (everything there is noise) must drop a lot.
	before := rmsDB(noisy, active, false)
	after := rmsDB(out, active, false)
	t.Logf("pause level: %.1f dBFS -> %.1f dBFS", before, after)
	if before-after < 15 {
		t.Errorf("pause noise reduced by only %.1f dB", before-after)
	}

	// Distance to the clean speech must shrink overall while speech stays intact.
	errIn := errDB(noisy, speech, active, true)
	errOut := errDB(out, speech, active, true)
	t.Logf("error vs clean speech in phrases: %.1f dB -> %.1f dB", errIn, errOut)
	if errOut > errIn-3 {
		t.Errorf("speech error did not improve enough: %.1f -> %.1f dB", errIn, errOut)
	}
	loss := rmsDB(speech, active, true) - rmsDB(out, active, true)
	t.Logf("speech level change: %+.2f dB", -loss)
	if math.Abs(loss) > 1.5 {
		t.Errorf("speech level changed by %.2f dB", -loss)
	}
}

func TestDenoiseTracksHum(t *testing.T) {
	speech, active := syntheticSpeech(15, 3)
	hum := make([]float32, len(speech))
	for i := range hum {
		ph := 2 * math.Pi * float64(i) / testRate
		hum[i] = float32(0.01 * (math.Sin(ph*50) + 0.5*math.Sin(ph*100) + 0.3*math.Sin(ph*150)))
	}
	noisy := mix(mix(speech, hum), whiteNoise(len(speech), 0.003, 4))
	b := mono(noisy)
	Spectral(b, SpectralConfig{AttenuationDB: 30, Overshoot: 1.2})
	before := rmsDB(noisy, active, false)
	after := rmsDB(b.Ch[0], active, false)
	t.Logf("hum+noise in pauses: %.1f -> %.1f dBFS", before, after)
	if before-after < 15 {
		t.Errorf("hum reduced by only %.1f dB", before-after)
	}
}

func TestAttenuationLimitIsRespected(t *testing.T) {
	speech, active := syntheticSpeech(15, 5)
	noisy := mix(speech, whiteNoise(len(speech), 0.01, 6))
	b := mono(noisy)
	Spectral(b, SpectralConfig{AttenuationDB: 12, Overshoot: 1})
	drop := rmsDB(noisy, active, false) - rmsDB(b.Ch[0], active, false)
	t.Logf("pause reduction with 12 dB limit: %.1f dB", drop)
	if drop > 14 || drop < 8 {
		t.Errorf("12 dB limit gave %.1f dB of reduction", drop)
	}
}

func TestDereverbShortensTail(t *testing.T) {
	speech, active := syntheticSpeech(12, 7)
	// Sparse exponentially decaying impulse response (RT60 = 0.5 s): a few
	// hundred random taps are enough to make a diffuse tail and keep the
	// direct convolution cheap.
	rng := rand.New(rand.NewSource(8))
	type tap struct {
		delay int
		gain  float32
	}
	var taps []tap
	for i := 0; i < 500; i++ {
		d := int(0.012*testRate) + rng.Intn(testRate/2)
		g := math.Pow(10, -3*float64(d)/(testRate/2)) * 0.05 * rng.NormFloat64()
		taps = append(taps, tap{d, float32(g)})
	}
	wet := append([]float32(nil), speech...)
	for i, s := range speech {
		if s == 0 {
			continue
		}
		for _, tp := range taps {
			if i+tp.delay < len(wet) {
				wet[i+tp.delay] += s * tp.gain
			}
		}
	}
	b := mono(wet)
	Spectral(b, SpectralConfig{Dereverb: 0.7, ReverbTime: 0.5})

	// Level in the pauses (pure reverb tail) should fall noticeably, while the
	// phrases keep their level.
	tailBefore := rmsDB(wet, active, false)
	tailAfter := rmsDB(b.Ch[0], active, false)
	speechDelta := rmsDB(b.Ch[0], active, true) - rmsDB(wet, active, true)
	t.Logf("tail %.1f -> %.1f dBFS, phrases %+.2f dB", tailBefore, tailAfter, speechDelta)
	if tailBefore-tailAfter < 4 {
		t.Errorf("reverb tail reduced by only %.1f dB", tailBefore-tailAfter)
	}
	if speechDelta < -3 {
		t.Errorf("phrases lost %.1f dB", -speechDelta)
	}
}

func TestVoiceProtectionRelaxesFloorOnlyInsideSpeech(t *testing.T) {
	speech, active := syntheticSpeech(20, 9)
	noisy := mix(speech, whiteNoise(len(speech), 0.01, 10))

	// The synthetic voice has nothing above 7 kHz, so what remains up there is
	// residual noise: a probe of how deep the floor is in each condition.
	hfLevel := func(x []float32, speaking bool) float64 {
		hp := dsp.Cascade{dsp.HighPass(testRate, 8000, 0.7), dsp.HighPass(testRate, 8000, 0.7)}
		y := make([]float32, len(x))
		for i, v := range x {
			y[i] = float32(hp.Process(float64(v)))
		}
		return rmsDB(y, active, speaking)
	}
	run := func(speechFloor float64) []float32 {
		b := mono(noisy)
		Spectral(b, SpectralConfig{AttenuationDB: 40, Overshoot: 1.2, SpeechFloorDB: speechFloor})
		return b.Ch[0]
	}
	hard, protected := run(0), run(10)

	inHard, inProt := hfLevel(hard, true), hfLevel(protected, true)
	paHard, paProt := hfLevel(hard, false), hfLevel(protected, false)
	t.Logf("HF residual inside phrases: %.1f -> %.1f dBFS; in pauses: %.1f -> %.1f dBFS", inHard, inProt, paHard, paProt)
	if inProt < inHard+8 {
		t.Errorf("floor not relaxed inside speech: %.1f vs %.1f dBFS", inProt, inHard)
	}
	// The relaxed floor releases over ~200 ms after a word, so the pauses are
	// not as deep as without protection, but they must stay clearly deeper
	// than the speech they separate.
	if inProt-paProt < 8 {
		t.Errorf("pauses only %.1f dB below the speech floor", inProt-paProt)
	}
}

func TestIsolatedClickIsRemovedButShortWordsStay(t *testing.T) {
	speech, active := syntheticSpeech(14, 13)
	noisy := mix(speech, whiteNoise(len(speech), 0.003, 14))
	rng := rand.New(rand.NewSource(15))

	// A 5 ms tick in the middle of the first pause (phrases run 0–1.6 s,
	// 2.5–4.1 s, …, pauses in between), and a 120 ms vowel-like burst in the
	// second pause, which must be kept.
	clickAt := int(2.0 * testRate)
	for i := 0; i < testRate/200; i++ {
		noisy[clickAt+i] += float32(0.3 * rng.NormFloat64())
	}
	wordAt := int(4.6 * testRate)
	for i := 0; i < testRate*120/1000; i++ {
		ph := 2 * math.Pi * float64(i) / testRate
		env := math.Sin(math.Pi * float64(i) / float64(testRate*120/1000))
		noisy[wordAt+i] += float32(0.1 * env * (math.Sin(ph*200) + 0.6*math.Sin(ph*400) + 0.4*math.Sin(ph*600)))
	}
	if active[clickAt] || active[wordAt] {
		t.Fatal("test signal: events must fall in pauses")
	}

	b := mono(noisy)
	Spectral(b, SpectralConfig{AttenuationDB: 30, Overshoot: 1.1, SpeechFloorDB: 8})

	peak := func(x []float32, from, n int) float64 {
		var p float64
		for _, v := range x[from : from+n] {
			p = math.Max(p, math.Abs(float64(v)))
		}
		return p
	}
	clickDrop := 20 * math.Log10(peak(b.Ch[0], clickAt-testRate/100, testRate/20)/peak(noisy, clickAt-testRate/100, testRate/20))
	wordDrop := 20 * math.Log10(peak(b.Ch[0], wordAt, testRate/8)/peak(noisy, wordAt, testRate/8))
	t.Logf("click peak %+.1f dB, 120 ms word peak %+.1f dB", clickDrop, wordDrop)
	if clickDrop > -15 {
		t.Errorf("click only reduced by %.1f dB", -clickDrop)
	}
	if wordDrop < -8 {
		t.Errorf("short word was attenuated by %.1f dB", -wordDrop)
	}
}

func TestFindClicksIgnoresWordsAndSustainedEvents(t *testing.T) {
	const n = 400
	thr := make([]float32, n/gridEvery+1)
	for i := range thr {
		thr[i] = 1
	}
	energy := make([]float32, n)
	for i := range energy {
		energy[i] = 1
	}
	raise := func(from, to int) {
		for i := from; i < to; i++ {
			energy[i] = 100
		}
	}
	raise(50, 52)   // 2 frames, isolated: a click
	raise(120, 140) // 20 frames: a word
	raise(200, 202) // 2 frames…
	raise(208, 230) // …followed 6 frames later by a word: a plosive, not a click
	raise(300, 303) // 3 frames, isolated: a click
	raise(396, 400) // 4 frames at the very end: too long

	got := findClicks(energy, thr)
	if len(got) != 2 {
		t.Fatalf("found %d clicks, want 2: %v", len(got), got)
	}
	for i, wantFrame := range []int{50, 300} {
		centre := wantFrame * strideSize
		if got[i][0] > centre || got[i][1] < centre {
			t.Errorf("click %d range %v does not cover frame %d", i, got[i], wantFrame)
		}
	}
}
