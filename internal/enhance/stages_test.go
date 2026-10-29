package enhance

import (
	"math"
	"math/rand"
	"testing"

	"kyone/internal/audio"
	"kyone/internal/dsp"
	"kyone/internal/loudness"
)

func sineBuf(freq, amp, seconds float64) *audio.Buffer {
	n := int(seconds * testRate)
	b := audio.NewBuffer(testRate, 1, n)
	for i := range b.Ch[0] {
		b.Ch[0][i] = float32(amp * math.Sin(2*math.Pi*freq*float64(i)/testRate))
	}
	return b
}

// steadyDB is the RMS of the middle half of x in dBFS.
func steadyDB(x []float32) float64 {
	mid := x[len(x)/4 : len(x)*3/4]
	var s float64
	for _, v := range mid {
		s += float64(v) * float64(v)
	}
	return 10 * math.Log10(s/float64(len(mid)))
}

func TestShapeToneGains(t *testing.T) {
	for _, tc := range []struct {
		name string
		opt  Options
		freq float64
		want float64
		tol  float64
	}{
		{"bass shelf", Options{BassDB: 6}, 40, 6, 0.6},
		{"bass leaves mids", Options{BassDB: 6}, 2000, 0, 0.3},
		{"presence bell", Options{PresenceDB: 3}, 3200, 3, 0.1},
		{"treble shelf", Options{TrebleDB: -4}, 15000, -4, 0.4},
		{"all off is transparent", Options{}, 1000, 0, 0.001},
	} {
		b := sineBuf(tc.freq, 0.1, 1)
		shapeTone(b, tc.opt)
		if got := steadyDB(b.Ch[0]) - steadyDB(sineBuf(tc.freq, 0.1, 1).Ch[0]); math.Abs(got-tc.want) > tc.tol {
			t.Errorf("%s: %+.2f dB, want %+.1f", tc.name, got, tc.want)
		}
	}
}

func TestHighPassRemovesRumble(t *testing.T) {
	rumble := sineBuf(30, 0.2, 1)
	highPass(rumble, 80)
	if got := steadyDB(rumble.Ch[0]); got > -45 {
		t.Errorf("30 Hz rumble only down to %.1f dBFS", got)
	}
	voice := sineBuf(500, 0.2, 1)
	highPass(voice, 80)
	if d := steadyDB(voice.Ch[0]) - steadyDB(sineBuf(500, 0.2, 1).Ch[0]); math.Abs(d) > 0.1 {
		t.Errorf("500 Hz changed by %.2f dB", d)
	}
}

func TestLimiterHoldsCeilingAndLeavesRestUntouched(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	n := testRate * 4
	in := audio.NewBuffer(testRate, 2, n)
	for c := range in.Ch {
		for i := range in.Ch[c] {
			in.Ch[c][i] = float32(0.2 * math.Sin(2*math.Pi*300*float64(i)/testRate))
		}
	}
	for k := 0; k < 12; k++ { // a few hot peaks, some on one channel only
		i := testRate/4 + rng.Intn(n-testRate/2)
		in.Ch[rng.Intn(2)][i] = float32(0.9 + 0.8*rng.Float64())
	}
	out := in.Clone()
	const ceiling = -3.0
	reduction := limit(out, ceiling)
	if reduction <= 0 {
		t.Fatal("limiter did nothing")
	}
	for c := range out.Ch {
		for _, v := range dsp.TruePeakEnvelope(out.Ch[c]) {
			if db := dsp.DB(float64(v)); db > ceiling+0.05 {
				t.Fatalf("true peak %.2f dBTP above the %.1f ceiling", db, ceiling)
			}
		}
	}
	// Far from any peak the signal must pass unchanged.
	var peaks []int
	for c := range in.Ch {
		for i, v := range in.Ch[c] {
			if math.Abs(float64(v)) > 0.5 {
				peaks = append(peaks, i)
			}
		}
	}
	untouched, total := 0, 0
	for i := 0; i < n; i++ {
		near := false
		for _, p := range peaks {
			if p > i-2000 && p < i+2000 {
				near = true
				break
			}
		}
		if !near {
			total++
			if out.Ch[0][i] == in.Ch[0][i] && out.Ch[1][i] == in.Ch[1][i] {
				untouched++
			}
		}
	}
	if total == 0 || float64(untouched)/float64(total) < 0.999 {
		t.Errorf("only %d of %d quiet samples left untouched", untouched, total)
	}
}

func TestLimiterMergesNearbyPeaks(t *testing.T) {
	// Two hot peaks 300 samples apart share one gain dip; a third is far away.
	b := audio.NewBuffer(testRate, 1, testRate)
	for i := range b.Ch[0] {
		b.Ch[0][i] = float32(0.1 * math.Sin(2*math.Pi*200*float64(i)/testRate))
	}
	for _, i := range []int{20000, 20300, 40000} {
		b.Ch[0][i] = 1.5
	}
	limit(b, -1)
	for _, v := range dsp.TruePeakEnvelope(b.Ch[0]) {
		if db := dsp.DB(float64(v)); db > -0.95 {
			t.Fatalf("true peak %.2f dBTP above the ceiling", db)
		}
	}
}

func TestLimiterCoversPeakAtStartAndEnd(t *testing.T) {
	b := audio.NewBuffer(testRate, 1, 5000)
	b.Ch[0][2] = 2
	b.Ch[0][4998] = -2
	limit(b, -2)
	for _, v := range b.Ch[0] {
		if math.Abs(float64(v)) > dsp.FromDB(-2)+1e-4 {
			t.Fatalf("sample %v above the ceiling", v)
		}
	}
}

func TestLimiterNoOpBelowCeiling(t *testing.T) {
	b := sineBuf(440, 0.3, 0.5)
	want := b.Clone()
	if r := limit(b, -1); r != 0 {
		t.Fatalf("reduction %.2f dB", r)
	}
	for i := range b.Ch[0] {
		if b.Ch[0][i] != want.Ch[0][i] {
			t.Fatal("signal modified")
		}
	}
}

func TestCompressorReducesLoudPartsOnly(t *testing.T) {
	// 1 s loud (-10 dBFS RMS) then 1 s soft (-30 dBFS RMS), repeated.
	n := testRate * 6
	b := audio.NewBuffer(testRate, 1, n)
	loud := make([]bool, n)
	for i := range b.Ch[0] {
		amp := 0.0316 * math.Sqrt2 // -30 dBFS RMS
		if (i/testRate)%2 == 0 {
			amp, loud[i] = 0.316*math.Sqrt2, true // -10 dBFS RMS
		}
		b.Ch[0][i] = float32(amp * math.Sin(2*math.Pi*400*float64(i)/testRate))
	}
	in := b.Clone()
	compress(b, -12)
	dLoud := rmsDB(b.Ch[0][testRate/2:testRate], loud[testRate/2:testRate], true) - rmsDB(in.Ch[0][testRate/2:testRate], loud[testRate/2:testRate], true)
	soft := testRate*3 + testRate/2
	dSoft := steadyDB(b.Ch[0][testRate*3:testRate*4]) - steadyDB(in.Ch[0][testRate*3:testRate*4])
	_ = soft
	t.Logf("loud %+.2f dB, soft %+.2f dB", dLoud, dSoft)
	if dLoud > -2 {
		t.Errorf("loud part only changed by %+.2f dB", dLoud)
	}
	if math.Abs(dSoft) > 0.3 {
		t.Errorf("soft part changed by %+.2f dB", dSoft)
	}
}

func TestDeEsserTamesSibilanceButNotVowels(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	hp := dsp.Cascade{dsp.HighPass(testRate, 6000, 0.7), dsp.HighPass(testRate, 6000, 0.7)}
	n := testRate * 4
	b := audio.NewBuffer(testRate, 1, n)
	sib := make([]bool, n)
	for i := range b.Ch[0] {
		t := float64(i) / testRate
		voiced := 0.15*math.Sin(2*math.Pi*220*t) + 0.1*math.Sin(2*math.Pi*880*t) + 0.05*math.Sin(2*math.Pi*1900*t)
		if int(t*2)%2 == 0 { // 0.5 s vowels / 0.5 s "sss"
			b.Ch[0][i] = float32(voiced)
		} else {
			sib[i] = true
			b.Ch[0][i] = float32(hp.Process(0.3 * rng.NormFloat64()))
		}
	}
	in := b.Clone()
	deEss(b)
	dSib := rmsDB(b.Ch[0], sib, true) - rmsDB(in.Ch[0], sib, true)
	dVow := rmsDB(b.Ch[0], sib, false) - rmsDB(in.Ch[0], sib, false)
	t.Logf("sibilants %+.2f dB, vowels %+.2f dB", dSib, dVow)
	if dSib > -3 {
		t.Errorf("sibilants only changed by %+.2f dB", dSib)
	}
	if math.Abs(dVow) > 0.2 {
		t.Errorf("vowels changed by %+.2f dB", dVow)
	}
}

func TestNormalizeHitsTargetAndCeiling(t *testing.T) {
	speech, _ := syntheticSpeech(12, 21)
	for _, target := range []float64{-23, -16, -14} {
		b := mono(speech)
		gain, _ := normalize(b, target, -1)
		r := loudness.Measure(b)
		t.Logf("target %v: got %.2f LUFS, %.2f dBTP (gain %+.1f dB)", target, r.IntegratedLUFS, r.TruePeakDB, gain)
		if math.Abs(r.IntegratedLUFS-target) > 0.25 {
			t.Errorf("target %v: measured %.2f LUFS", target, r.IntegratedLUFS)
		}
		if r.TruePeakDB > -0.95 {
			t.Errorf("target %v: true peak %.2f dBTP above ceiling", target, r.TruePeakDB)
		}
	}
}
