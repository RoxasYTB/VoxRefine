package enhance

import (
	"math"
	"testing"

	"kyone/internal/dsp"
	"kyone/internal/loudness"
)

func levelsOf(x []float32) (noise, speech float64) {
	return loudness.Levels(mono(x))
}

func TestClarityLiftsPresenceInSpeechOnly(t *testing.T) {
	speech, active := syntheticSpeech(20, 81)
	x := mix(speech, whiteNoise(len(speech), 0.002, 82))
	noise, spk := levelsOf(x)

	b := mono(x)
	clarity(b, 1, noise, spk)
	presence := bandRMS(b.Ch[0], 2500, 4000, active, true) - bandRMS(x, 2500, 4000, active, true)
	body := bandRMS(b.Ch[0], 300, 1200, active, true) - bandRMS(x, 300, 1200, active, true)
	pauses := rmsDB(b.Ch[0], active, false) - rmsDB(x, active, false)
	t.Logf("presence %+.1f dB, body %+.1f dB, pauses %+.2f dB", presence, body, pauses)
	// The nominal +5 dB is gated by the syllable-by-syllable activity, so the
	// average lift over speech is a fraction of it.
	if presence < 1 || presence > 5 {
		t.Errorf("presence band moved by %+.1f dB, want +1…+5", presence)
	}
	if math.Abs(body) > 0.7 {
		t.Errorf("body band moved by %+.1f dB", body)
	}
	if math.Abs(pauses) > 0.5 {
		t.Errorf("pauses changed by %+.2f dB: the lift must be gated by speech", pauses)
	}
}

func TestClarityStrengthZeroIsTransparent(t *testing.T) {
	speech, _ := syntheticSpeech(4, 83)
	b := mono(speech)
	want := b.Clone()
	noise, spk := levelsOf(speech)
	clarity(b, 0, noise, spk)
	for i := range want.Ch[0] {
		if b.Ch[0][i] != want.Ch[0][i] {
			t.Fatal("signal changed at zero strength")
		}
	}
}

func TestClarityLeavesSibilantsAlone(t *testing.T) {
	// Vowels at -20 dBFS alternating with 6–10 kHz noise bursts of similar
	// level: the bursts have no voiced energy, so the air lift stays shut.
	hp := dsp.Cascade{dsp.HighPass(testRate, 6000, 0.7), dsp.HighPass(testRate, 6000, 0.7)}
	n := testRate * 6
	x := make([]float32, n)
	sib := make([]bool, n)
	noiseSrc := whiteNoise(n, 0.15, 84)
	for i := range x {
		sec := float64(i) / testRate
		if int(sec*2)%2 == 0 {
			x[i] = float32(0.1*math.Sin(2*math.Pi*220*sec) + 0.05*math.Sin(2*math.Pi*880*sec))
		} else {
			sib[i] = true
			x[i] = float32(hp.Process(float64(noiseSrc[i])))
		}
	}
	noise, spk := levelsOf(x)
	b := mono(x)
	clarity(b, 1, noise, spk)
	d := bandRMS(b.Ch[0], 6500, 12000, sib, true) - bandRMS(x, 6500, 12000, sib, true)
	t.Logf("sibilant band moved by %+.2f dB", d)
	if d > 1 {
		t.Errorf("sibilants brightened by %+.1f dB", d)
	}
}

func TestClaritySkipsNoisyRecordingsWithNoHeadroom(t *testing.T) {
	// Speech barely above the noise: nothing to lift safely.
	speech, _ := syntheticSpeech(8, 85)
	x := mix(speech, whiteNoise(len(speech), 0.05, 86))
	b := mono(x)
	want := b.Clone()
	noise, spk := levelsOf(x)
	if spk-noise >= 10 {
		t.Skip("signal-to-noise too good for this test")
	}
	clarity(b, 1, noise, spk)
	for i := range want.Ch[0] {
		if b.Ch[0][i] != want.Ch[0][i] {
			t.Fatal("clarity ran with less than 10 dB of headroom")
		}
	}
}
