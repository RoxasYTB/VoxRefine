package enhance

import (
	"math"
	"strings"
	"testing"

	"kyone/internal/audio"
	"kyone/internal/loudness"
)

func TestRunEndToEnd(t *testing.T) {
	speech, active := syntheticSpeech(15, 31)
	noisy := mix(speech, whiteNoise(len(speech), 0.01, 32))
	in := audio.Resample(mono(noisy), 44100) // exercise the resampler too

	var stages []string
	out, rep, err := Run(in, DefaultOptions(), func(s string) { stages = append(stages, s) })
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("stages: %s\n%s", strings.Join(stages, ", "), rep.Summary())

	if out.Rate != SampleRate || out.Channels() != 1 {
		t.Fatalf("output is %d Hz, %d channel(s)", out.Rate, out.Channels())
	}
	if want := int(math.Round(15 * SampleRate)); math.Abs(float64(out.Frames()-want)) > 2 {
		t.Errorf("output has %d frames, want ≈%d", out.Frames(), want)
	}
	if math.Abs(rep.After.IntegratedLUFS-(-16)) > 0.3 {
		t.Errorf("loudness %.2f LUFS, want -16", rep.After.IntegratedLUFS)
	}
	if rep.After.TruePeakDB > -0.9 {
		t.Errorf("true peak %.2f dBTP above the -1 ceiling", rep.After.TruePeakDB)
	}
	if gain := rep.After.SNR() - rep.Before.SNR(); gain < 8 {
		t.Errorf("estimated SNR improved by only %.1f dB", gain)
	}
	if rep.After.Clipped != 0 {
		t.Errorf("%d clipped samples", rep.After.Clipped)
	}
	_ = active
}

func TestRunStereoKeepsImageAndMonoOptionFolds(t *testing.T) {
	speech, _ := syntheticSpeech(8, 41)
	noisy := mix(speech, whiteNoise(len(speech), 0.008, 42))
	left := noisy
	right := make([]float32, len(noisy))
	for i, v := range noisy {
		right[i] = 0.5 * v
	}
	in := &audio.Buffer{Rate: testRate, Ch: [][]float32{left, right}}

	out, _, err := Run(in.Clone(), DefaultOptions(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if out.Channels() != 2 {
		t.Fatalf("got %d channels", out.Channels())
	}
	// Linked processing keeps the 2:1 level ratio between channels.
	var l, r float64
	for i := range out.Ch[0] {
		l += float64(out.Ch[0][i]) * float64(out.Ch[0][i])
		r += float64(out.Ch[1][i]) * float64(out.Ch[1][i])
	}
	if ratio := 10 * math.Log10(l/r); math.Abs(ratio-6.02) > 0.3 {
		t.Errorf("channel balance %.2f dB, want ≈6.02", ratio)
	}

	opt := DefaultOptions()
	opt.Mono = true
	m, _, err := Run(in, opt, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m.Channels() != 1 {
		t.Fatalf("mono option gave %d channels", m.Channels())
	}
}

func TestRunWithEverythingOffIsNearlyTransparent(t *testing.T) {
	speech, _ := syntheticSpeech(4, 51)
	in := mono(speech)
	out, _, err := Run(in.Clone(), Options{CeilingDB: -1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Peaks are below the ceiling, so nothing may change at all.
	for i, v := range in.Ch[0] {
		if out.Ch[0][i] != v {
			t.Fatalf("sample %d changed: %v -> %v", i, v, out.Ch[0][i])
		}
	}
}

func TestRunRejectsSilenceAndTinyInput(t *testing.T) {
	if _, _, err := Run(audio.NewBuffer(48000, 1, 48000), DefaultOptions(), nil); err == nil {
		t.Error("silence should be rejected")
	}
	if _, _, err := Run(audio.NewBuffer(48000, 1, 100), DefaultOptions(), nil); err == nil {
		t.Error("tiny input should be rejected")
	}
}

func TestRunDoesNotAmplifyNoiseOnlyInputIntoNoise(t *testing.T) {
	// A pure noise file has nothing to normalise towards; the gain is capped.
	noise := mono(whiteNoise(testRate*5, 0.0001, 61))
	out, rep, err := Run(noise, DefaultOptions(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if rep.GainDB > maxNormalizeGainDB+0.01 {
		t.Errorf("gain %.1f dB exceeds the cap", rep.GainDB)
	}
	if got := loudness.Measure(out).SamplePeakDB; got > 0 {
		t.Errorf("peak %.1f dBFS", got)
	}
}

func TestOptionsValidationAndProfiles(t *testing.T) {
	if _, err := OptionsFor("nope"); err == nil {
		t.Error("unknown profile accepted")
	}
	prev := -1.0
	for _, p := range Profiles() {
		o, err := OptionsFor(p.Name)
		if err != nil || o.AttenuationLimitDB < prev {
			t.Errorf("profile %s: %v, limit %.0f below the previous profile's %.0f", p.Name, err, o.AttenuationLimitDB, prev)
		}
		prev = o.AttenuationLimitDB
		if err := o.Validate(); err != nil {
			t.Errorf("profile %s invalid: %v", p.Name, err)
		}
	}
	bad := DefaultOptions()
	bad.AttenuationLimitDB = 101
	if bad.Validate() == nil {
		t.Error("101 dB limit accepted")
	}
	bad = DefaultOptions()
	bad.BassDB = 20
	if bad.Validate() == nil {
		t.Error("20 dB bass accepted")
	}
	if a, b := OvershootFor(0), OvershootFor(100); a != 1.0 || b < 1.2 {
		t.Errorf("overshoot interpolation: %v %v", a, b)
	}
}

func TestParseEQBand(t *testing.T) {
	for spec, want := range map[string]EQBand{
		"low:200:+9":       {"low", 200, 9, 0},
		"high:6300:-3.5":   {"high", 6300, -3.5, 0},
		"peak:3000:2":      {"peak", 3000, 2, 0},
		"peak:3000:-2:1.4": {"peak", 3000, -2, 1.4},
	} {
		got, err := ParseEQBand(spec)
		if err != nil || got != want {
			t.Errorf("ParseEQBand(%q) = %+v, %v; want %+v", spec, got, err, want)
		}
		if again, err := ParseEQBand(got.String()); err != nil || again != got {
			t.Errorf("String round trip of %q gave %+v, %v", spec, again, err)
		}
	}
	for _, bad := range []string{"", "low", "low:200", "mid:200:3", "low:abc:3", "low:200:3:1", "peak:1:2:3:4"} {
		if _, err := ParseEQBand(bad); err == nil {
			t.Errorf("ParseEQBand(%q) accepted", bad)
		}
	}
	o := DefaultOptions()
	o.EQ = []EQBand{{"low", 5, 3, 0}}
	if o.Validate() == nil {
		t.Error("5 Hz EQ band accepted")
	}
	o.EQ = []EQBand{{"peak", 1000, 30, 0}}
	if o.Validate() == nil {
		t.Error("+30 dB EQ band accepted")
	}
}

func TestEQBandsShapeTheSpectrum(t *testing.T) {
	opt := Options{EQ: []EQBand{{"low", 200, 12, 0}, {"high", 2000, -6, 0}, {"peak", 1000, 4, 1}}}
	for _, tc := range []struct {
		freq, want, tol float64
	}{{40, 12, 0.8}, {1000, 4 + 12*0.05 - 6*0.4, 3.5}, {12000, -6, 0.5}} {
		b := sineBuf(tc.freq, 0.1, 1)
		shapeTone(b, opt)
		got := steadyDB(b.Ch[0]) - steadyDB(sineBuf(tc.freq, 0.1, 1).Ch[0])
		if math.Abs(got-tc.want) > tc.tol {
			t.Errorf("%g Hz: %+.1f dB, want ≈%+.1f", tc.freq, got, tc.want)
		}
	}
}
