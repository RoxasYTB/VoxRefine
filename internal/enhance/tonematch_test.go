package enhance

import (
	"math"
	"testing"

	"kyone/internal/dsp"
)

// bandRMS is the RMS of x between lo and hi Hz over the samples where sel == want.
func bandRMS(x []float32, lo, hi float64, sel []bool, want bool) float64 {
	c := dsp.Cascade{
		dsp.HighPass(testRate, lo, 0.707), dsp.HighPass(testRate, lo, 0.707),
		dsp.LowPass(testRate, hi, 0.707), dsp.LowPass(testRate, hi, 0.707),
	}
	y := make([]float32, len(x))
	for i, v := range x {
		y[i] = float32(c.Process(float64(v)))
	}
	return rmsDB(y, sel, want)
}

// thinVoice is the synthetic voice through a 2nd-order high-pass at 400 Hz,
// like a laptop microphone with no bass, plus a faint noise floor.
func thinVoice(seconds float64, seed int64) (x []float32, active []bool) {
	speech, active := syntheticSpeech(seconds, seed)
	hp := dsp.Cascade{dsp.HighPass(testRate, 400, 0.707), dsp.HighPass(testRate, 400, 0.707)}
	x = make([]float32, len(speech))
	for i, v := range speech {
		x[i] = float32(hp.Process(float64(v)))
	}
	return mix(x, whiteNoise(len(x), 0.0005, seed+100)), active
}

func TestToneMatchRestoresBassOfAThinVoice(t *testing.T) {
	x, active := thinVoice(20, 71)
	b := mono(x)
	raw := probeTone(b)
	if raw == nil {
		t.Fatal("no probe")
	}
	before := bandRMS(x, 100, 250, active, true) - bandRMS(x, 600, 1500, active, true)
	corr := toneMatch(b, raw, 1)
	if corr == nil {
		t.Fatal("tone match skipped")
	}
	after := bandRMS(b.Ch[0], 100, 250, active, true) - bandRMS(b.Ch[0], 600, 1500, active, true)
	t.Logf("bass over mids: %.1f dB -> %.1f dB", before, after)
	if after-before < 5 {
		t.Errorf("bass only restored by %.1f dB", after-before)
	}
	// Every band correction stays within the safety limits.
	for band, c := range corr {
		if c > maxBoost(toneCentres[band])+1e-9 || c < toneMinBoost-1e-9 {
			t.Errorf("%.0f Hz corrected by %+.1f dB, outside the limits", toneCentres[band], c)
		}
	}
}

func TestToneMatchStrengthScalesTheCorrection(t *testing.T) {
	x, _ := thinVoice(20, 72)
	raw := probeTone(mono(x))
	full := toneMatch(mono(x), raw, 1)
	half := toneMatch(mono(x), raw, 0.5)
	for b := range full {
		if math.Abs(half[b]-full[b]/2) > 1e-9 {
			t.Fatalf("band %d: half %.3f, full %.3f", b, half[b], full[b])
		}
	}
	if toneMatch(mono(x), raw, 0) != nil {
		t.Error("zero strength should skip the stage")
	}
}

func TestToneMatchNeverBoostsWhatTheVoiceLacks(t *testing.T) {
	// A voice band-limited at 4 kHz has nothing above: the studio curve's
	// presence region must not be filled with amplified noise.
	speech, _ := syntheticSpeech(20, 73)
	lp := dsp.Cascade{dsp.LowPass(testRate, 3500, 0.707), dsp.LowPass(testRate, 3500, 0.707), dsp.LowPass(testRate, 3500, 0.707)}
	x := make([]float32, len(speech))
	for i, v := range speech {
		x[i] = float32(lp.Process(float64(v)))
	}
	x = mix(x, whiteNoise(len(x), 0.0003, 74))
	b := mono(x)
	corr := toneMatch(b, probeTone(b), 1)
	for band, c := range corr {
		if toneCentres[band] > 6000 && c > 1 {
			t.Errorf("%.0f Hz boosted by %+.1f dB although the voice has no content there", toneCentres[band], c)
		}
		if toneCentres[band] > 8000 && c > 4+1e-9 {
			t.Errorf("%.0f Hz boosted by %+.1f dB, above the air limit", toneCentres[band], c)
		}
	}
}

func TestToneMatchSkipsWhenThereIsNothingToJudge(t *testing.T) {
	short := mono(whiteNoise(testRate/4, 0.01, 75))
	if probeTone(short) != nil {
		t.Error("probe of a 250 ms clip should be nil")
	}
	x, _ := thinVoice(8, 76)
	b := mono(x)
	before := b.Clone()
	if corr := toneMatch(b, nil, 1); corr != nil {
		t.Error("tone match without a probe should be skipped")
	}
	for i := range before.Ch[0] {
		if b.Ch[0][i] != before.Ch[0][i] {
			t.Fatal("signal changed although the stage was skipped")
		}
	}
}

func TestToneSummary(t *testing.T) {
	r := &Report{}
	if r.ToneSummary() != "" {
		t.Error("summary without tone match should be empty")
	}
	r.ToneMatchDB = make([]float64, len(toneCentres))
	if got := r.ToneSummary(); got != "already close to the studio curve" {
		t.Errorf("flat curve: %q", got)
	}
	r.ToneMatchDB[2], r.ToneMatchDB[10] = 9, -3
	if got := r.ToneSummary(); got != "boost up to +9.0 dB around 159 Hz, cut up to -3.0 dB around 1008 Hz" {
		t.Errorf("got %q", got)
	}
}
