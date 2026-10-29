package enhance

import (
	"math"
	"testing"
	"time"

	"kyone/internal/loudness"
)

// runLive pushes x through a Live in blocks of the given size.
func runLive(t *testing.T, opt Options, levelDB float64, x []float32, block int) (*Live, []float32) {
	t.Helper()
	l, err := NewLive(opt, levelDB)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(x))
	for s := 0; s < len(x); s += block {
		e := min(s+block, len(x))
		l.Process(x[s:e], out[s:e])
	}
	return l, out
}

// aligned undoes the output delay: it returns in[:n-d] and out[d:] so that
// sample i of both refers to the same moment.
func aligned(l *Live, in, out []float32) (x, y []float32) {
	d := l.Latency()
	return in[:len(in)-d], out[d:]
}

func liveOptions() Options {
	o := DefaultOptions()
	o.AttenuationLimitDB = 30
	o.Overshoot = 1.1
	return o
}

func TestLiveIsIndependentOfBlockSize(t *testing.T) {
	speech, _ := syntheticSpeech(8, 91)
	x := mix(speech, whiteNoise(len(speech), 0.004, 92))
	_, ref := runLive(t, liveOptions(), 0, x, 480)
	for _, block := range []int{1, 97, 256, 1000, 4096} {
		_, got := runLive(t, liveOptions(), 0, x, block)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("block %d: sample %d differs (%v vs %v)", block, i, got[i], ref[i])
			}
		}
	}
}

func TestLiveLatencyMatchesTheReport(t *testing.T) {
	speech, _ := syntheticSpeech(6, 93)
	x := mix(speech, whiteNoise(len(speech), 0.002, 94))
	l, out := runLive(t, liveOptions(), 0, x, 480)

	// Find the delay that best aligns the output with the input.
	best, bestV := 0, math.Inf(-1)
	from, to := testRate*3, testRate*5
	for lag := 0; lag < 3000; lag++ {
		var v float64
		for i := from; i < to; i += 3 {
			v += float64(out[i]) * float64(x[i-lag])
		}
		if v > bestV {
			best, bestV = lag, v
		}
	}
	t.Logf("reported latency %d samples (%.1f ms), measured %d", l.Latency(), 1000*float64(l.Latency())/testRate, best)
	if d := best - l.Latency(); d < -12 || d > 12 { // filters add a little group delay
		t.Errorf("measured delay %d, reported %d", best, l.Latency())
	}
}

func TestLiveDenoisesAfterWarmUp(t *testing.T) {
	speech, active := syntheticSpeech(25, 95)
	x := mix(speech, whiteNoise(len(speech), 0.01, 96))
	opt := liveOptions()
	opt.Normalize = false
	l, y := runLive(t, opt, 0, x, 480)
	x, out := aligned(l, x, y)

	// Compare after a 5 s warm-up.
	from := testRate * 5
	pause := func(s []float32) float64 { return rmsDB(s[from:], active[from:len(s)], false) }
	speechDB := func(s []float32) float64 { return rmsDB(s[from:], active[from:len(s)], true) }
	reduction := pause(x) - pause(out)
	change := speechDB(out) - speechDB(x)
	t.Logf("pauses %.1f -> %.1f dBFS (%.1f dB), speech %+.2f dB", pause(x), pause(out), reduction, change)
	if reduction < 12 {
		t.Errorf("pauses reduced by only %.1f dB", reduction)
	}
	if math.Abs(change) > 3 {
		t.Errorf("speech level changed by %+.1f dB", change)
	}
}

func TestLiveFollowsHum(t *testing.T) {
	speech, active := syntheticSpeech(20, 97)
	x := make([]float32, len(speech))
	for i := range x {
		ph := 2 * math.Pi * float64(i) / testRate
		x[i] = speech[i] + float32(0.01*(math.Sin(ph*50)+0.5*math.Sin(ph*100)))
	}
	x = mix(x, whiteNoise(len(x), 0.002, 98))
	opt := liveOptions()
	opt.Normalize = false
	opt.HighPassHz = 0
	l, y := runLive(t, opt, 0, x, 480)
	x, out := aligned(l, x, y)
	from := testRate * 5
	d := rmsDB(x[from:], active[from:len(x)], false) - rmsDB(out[from:], active[from:len(x)], false)
	t.Logf("hum and noise in pauses reduced by %.1f dB", d)
	if d < 12 {
		t.Errorf("hum reduced by only %.1f dB", d)
	}
}

func TestLiveLimiterHoldsTheCeiling(t *testing.T) {
	speech, _ := syntheticSpeech(6, 99)
	x := make([]float32, len(speech))
	for i, v := range speech {
		x[i] = 9 * v // far too hot
	}
	x[20000] = 3
	x[20001] = -3
	opt := Options{CeilingDB: -3}
	l, out := runLive(t, opt, 0, x, 333)
	limit := float32(dsp_fromDB(-3)) + 1e-4
	var peak float32
	for _, v := range out {
		peak = max(peak, float32(math.Abs(float64(v))))
	}
	t.Logf("peak %.4f, ceiling %.4f, latency %d samples", peak, limit, l.Latency())
	if peak > limit {
		t.Errorf("output peak %.4f above the ceiling %.4f", peak, limit)
	}
	// Quiet material must pass untouched (apart from the delay).
	q := mono(whiteNoise(testRate, 0.01, 100)).Ch[0]
	l2, out2 := runLive(t, opt, 0, q, 480)
	d := l2.Latency()
	for i := d + 10; i < len(q); i++ {
		if out2[i] != q[i-d] {
			t.Fatalf("quiet sample %d changed: %v vs %v", i, out2[i], q[i-d])
		}
	}
}

func dsp_fromDB(db float64) float64 { return math.Pow(10, db/20) }

func TestLiveLevelerReachesTheTargetAndNeverAmplifiesSilence(t *testing.T) {
	speech, _ := syntheticSpeech(40, 101)
	opt := Options{Normalize: true, CeilingDB: -1}
	level := func(x []float32) float64 {
		_, sp := loudness.Levels(mono(x))
		return sp
	}
	for _, scale := range []float32{0.4, 6} { // quiet (needs +12 dB) and hot
		x := make([]float32, len(speech))
		for i, v := range speech {
			x[i] = v * scale
		}
		_, out := runLive(t, opt, -20, x, 480)
		got := level(out[testRate*25:])
		t.Logf("input speech %.1f dBFS -> %.1f dBFS (target -20)", level(x[testRate*25:]), got)
		if math.Abs(got+20) > 4 {
			t.Errorf("scale %v: speech level %.1f dBFS, want ≈ -20", scale, got)
		}
	}

	// Noise only: the leveler must not pump it up.
	noise := whiteNoise(testRate*30, 0.003, 102)
	_, out := runLive(t, opt, -20, noise, 480)
	in, got := rmsDB(noise[testRate*10:], make([]bool, len(noise)-testRate*10), false), rmsDB(out[testRate*10:], make([]bool, len(out)-testRate*10), false)
	t.Logf("noise-only: %.1f -> %.1f dBFS", in, got)
	if got > in+3 {
		t.Errorf("noise raised by %.1f dB", got-in)
	}
}

func TestLiveRunsFarFasterThanRealTime(t *testing.T) {
	speech, _ := syntheticSpeech(10, 103)
	x := mix(speech, whiteNoise(len(speech), 0.005, 104))
	opt, _ := OptionsFor("studio")
	l, err := NewLive(opt, 0)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, 480)
	start := time.Now()
	for s := 0; s+480 <= len(x); s += 480 {
		l.Process(x[s:s+480], out)
	}
	rtf := time.Since(start).Seconds() / (float64(len(x)) / testRate)
	t.Logf("real-time factor %.3f (%.1f%% of one core), latency %.1f ms", rtf, 100*rtf, 1000*float64(l.Latency())/testRate)
	if rtf > 0.5 {
		t.Errorf("real-time factor %.2f is too slow for live use", rtf)
	}
}

func TestLiveStartsSilentAndRejectsBadSettings(t *testing.T) {
	x := whiteNoise(testRate, 0.05, 105)
	l, out := runLive(t, liveOptions(), 0, x, 480)
	for i := 0; i < l.Latency()-1; i++ {
		if math.Abs(float64(out[i])) > 1e-3 {
			t.Fatalf("sample %d before the latency elapsed is %v", i, out[i])
		}
	}
	if _, err := NewLive(liveOptions(), -3); err == nil {
		t.Error("speech level -3 dBFS accepted")
	}
	bad := liveOptions()
	bad.AttenuationLimitDB = 200
	if _, err := NewLive(bad, 0); err == nil {
		t.Error("invalid options accepted")
	}
}
