package enhance

import (
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"

	"kyone/internal/dsp"
)

// fakeNeural halves the signal and delays it by `delay` samples, in frames.
type fakeStream struct {
	frame, delay int
	hist         []float32
	closed       bool
}

func newFakeStream(frame, delay int) *fakeStream {
	return &fakeStream{frame: frame, delay: delay, hist: make([]float32, delay)}
}

func (f *fakeStream) FrameLength() int { return f.frame }
func (f *fakeStream) Latency() int     { return f.delay }
func (f *fakeStream) Close()           { f.closed = true }
func (f *fakeStream) Process(in, out []float32) {
	f.hist = append(f.hist, in...)
	for i := range out {
		out[i] = 0.5 * f.hist[i]
	}
	f.hist = f.hist[len(in):]
}

func fakeNeural(stream *fakeStream) *Neural {
	return &Neural{
		Name: "fake",
		File: func(x []float32, attenDB float64) ([]float32, error) {
			y := make([]float32, len(x))
			for i, v := range x {
				y[i] = 0.5 * v
			}
			return y, nil
		},
		Open: func(attenDB float64) (StreamDenoiser, error) { return stream, nil },
	}
}

func TestMinPhaseFIRFollowsTheCurve(t *testing.T) {
	// A realistic tone curve: +12 dB of warmth falling to -4 dB at the top.
	corr := make([]float64, len(toneCentres))
	for b, fc := range toneCentres {
		corr[b] = 12 - 16*math.Min(math.Log2(fc/100)/7, 1)
	}
	gains := toneGains(corr, SampleRate)
	h := minPhaseFIR(gains, toneFIRTaps)

	// Magnitude response against the design curve.
	var worst float64
	for _, k := range []int{6, 12, 25, 50, 100, 200, 300, 400} {
		var resp complex128
		w := 2 * math.Pi * float64(k) / fftSize
		for n, c := range h {
			resp += complex(c, 0) * cmplx.Exp(complex(0, -w*float64(n)))
		}
		d := math.Abs(dsp.DB(cmplx.Abs(resp)) - dsp.DB(gains[k]))
		worst = math.Max(worst, d)
	}
	t.Logf("worst deviation from the design curve: %.2f dB", worst)
	if worst > 1.0 {
		t.Errorf("FIR deviates by %.2f dB from the curve", worst)
	}

	// Minimum phase: the energy sits at the start, so the filter adds no
	// noticeable delay.
	var head, total float64
	for n, c := range h {
		total += c * c
		if n < toneFIRTaps/4 {
			head += c * c
		}
	}
	if head/total < 0.97 {
		t.Errorf("only %.1f%% of the energy in the first quarter of the taps", 100*head/total)
	}
}

func TestAdaptiveFIRMatchesConvolutionAndCrossfadesSmoothly(t *testing.T) {
	f := newAdaptiveFIR(16, SampleRate)
	x := whiteNoise(4000, 0.3, 111)
	// Identity until a filter is set.
	for i := 0; i < 100; i++ {
		if got := f.process(float64(x[i])); got != float64(x[i]) {
			t.Fatalf("not transparent at %d", i)
		}
	}
	h := make([]float64, 16)
	h[0], h[3] = 0.5, 0.25
	f.set(h)
	var maxJump float64
	prev := 0.0
	for i := 100; i < 4000; i++ {
		y := f.process(float64(x[i]))
		if i >= 100+f.fadeLen+16 { // after the crossfade: exact convolution
			want := 0.5*float64(x[i]) + 0.25*float64(x[i-3])
			if math.Abs(y-want) > 1e-12 {
				t.Fatalf("sample %d: %v want %v", i, y, want)
			}
		}
		if i > 100 {
			maxJump = math.Max(maxJump, math.Abs(y-prev))
		}
		prev = y
	}
	if maxJump > 1.5 {
		t.Errorf("output jumped by %.2f during the crossfade", maxJump)
	}
}

func TestOfflineNeuralReplacesTheDenoiserAndKeepsTheRestOfTheChain(t *testing.T) {
	speech, _ := syntheticSpeech(8, 121)
	noisy := mix(speech, whiteNoise(len(speech), 0.01, 122))

	var stages []string
	opt := DefaultOptions()
	opt.Neural = fakeNeural(nil)
	_, rep, err := Run(mono(noisy), opt, func(s string) { stages = append(stages, s) })
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(stages, ",")
	t.Logf("stages: %s", joined)
	if !strings.Contains(joined, "denoise (fake)") {
		t.Error("the neural stage did not run")
	}
	if strings.Contains(joined, "denoise / dereverb") && opt.Dereverb == 0 {
		t.Error("the native denoiser ran as well")
	}
	if !strings.Contains(joined, "loudness") || math.Abs(rep.After.IntegratedLUFS+16) > 0.5 {
		t.Errorf("the rest of the chain did not run (%.1f LUFS)", rep.After.IntegratedLUFS)
	}

	// 0 dB limit means "no noise reduction": the neural stage is skipped.
	opt.AttenuationLimitDB = 0
	stages = nil
	if _, _, err := Run(mono(noisy), opt, func(s string) { stages = append(stages, s) }); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.Join(stages, ","), "fake") {
		t.Error("neural stage ran with a 0 dB limit")
	}

	// Errors surface instead of being swallowed.
	opt = DefaultOptions()
	opt.Neural = &Neural{Name: "broken", File: func([]float32, float64) ([]float32, error) { return nil, errors.New("boom") }}
	if _, _, err := Run(mono(noisy), opt, nil); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("error not reported: %v", err)
	}
	opt.Neural = &Neural{Name: "short", File: func(x []float32, _ float64) ([]float32, error) { return x[:10], nil }}
	if _, _, err := Run(mono(noisy), opt, nil); err == nil {
		t.Error("a result of the wrong length was accepted")
	}
	opt.Neural = &Neural{Name: "live-only"}
	if _, _, err := Run(mono(noisy), opt, nil); err == nil {
		t.Error("a denoiser without a file mode was accepted")
	}
}

func TestLiveNeuralPathRoutesThroughTheStreamDenoiser(t *testing.T) {
	speech, _ := syntheticSpeech(6, 123)
	x := mix(speech, whiteNoise(len(speech), 0.002, 124))
	stream := newFakeStream(480, 480)
	opt := Options{Neural: fakeNeural(stream), AttenuationLimitDB: 30, CeilingDB: -1}

	l, err := NewLive(opt, 0)
	if err != nil {
		t.Fatal(err)
	}
	if want := 480 + 480 + l.lim.latency(); l.Latency() != want {
		t.Errorf("latency %d, want %d (frame + denoiser + limiter)", l.Latency(), want)
	}
	out := make([]float32, len(x))
	for s := 0; s < len(x); s += 333 {
		e := min(s+333, len(x))
		l.Process(x[s:e], out[s:e])
	}
	// The fake halves and delays the signal; the limiter passes it unchanged.
	d := l.Latency()
	var worst float64
	for i := d + 2000; i < len(x); i++ {
		worst = math.Max(worst, math.Abs(float64(out[i])-0.5*float64(x[i-d])))
	}
	if worst > 1e-5 {
		t.Errorf("output differs from the expected half-level, delayed signal by %g", worst)
	}

	l.Close()
	if !stream.closed {
		t.Error("Close did not release the stream denoiser")
	}

	// Failure paths.
	if _, err := NewLive(Options{Neural: &Neural{Name: "offline-only"}, AttenuationLimitDB: 30, CeilingDB: -1}, 0); err == nil {
		t.Error("a denoiser without a live mode was accepted")
	}
	bad := &Neural{Name: "bad", Open: func(float64) (StreamDenoiser, error) { return nil, errors.New("no model") }}
	if _, err := NewLive(Options{Neural: bad, AttenuationLimitDB: 30, CeilingDB: -1}, 0); err == nil || !strings.Contains(err.Error(), "no model") {
		t.Errorf("open error not reported: %v", err)
	}
}

func TestSetBlockSizeTrimsTheStartUpDelay(t *testing.T) {
	mk := func() *Live {
		l, err := NewLive(Options{Neural: fakeNeural(newFakeStream(480, 480)), AttenuationLimitDB: 30, CeilingDB: -1}, 0)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	l := mk()
	before := l.Latency()
	l.SetBlockSize(480) // blocks of exactly one frame: no alignment slack needed
	if l.Latency() != before-480 {
		t.Errorf("latency %d, want %d", l.Latency(), before-480)
	}
	// Blocks of exactly one frame are answered in full without any slack.
	x := whiteNoise(480*50, 0.1, 125)
	out := make([]float32, len(x))
	for s := 0; s < len(x); s += 480 {
		l.Process(x[s:s+480], out[s:s+480])
	}
	d := l.Latency()
	for i := d + 10; i < len(x); i++ {
		if math.Abs(float64(out[i])-0.5*float64(x[i-d])) > 1e-6 {
			t.Fatalf("sample %d wrong after trimming (latency %d)", i, d)
		}
	}

	// An awkward block size keeps part of the slack, and calling after the
	// first block changes nothing.
	l2 := mk()
	b2 := l2.Latency()
	l2.SetBlockSize(100) // gcd(100, 480) = 20 → 460 samples of slack
	if got := l2.Latency(); got != b2-20 {
		t.Errorf("latency %d, want %d", got, b2-20)
	}
	l3 := mk()
	l3.Process(make([]float32, 10), make([]float32, 10))
	before3 := l3.Latency()
	l3.SetBlockSize(480)
	if l3.Latency() != before3 {
		t.Error("SetBlockSize after Process must do nothing")
	}
}

func TestLiveNeuralWithToneMatchLearnsAndStaysDelayFree(t *testing.T) {
	x, _ := thinVoice(30, 131)
	opt := Options{Neural: fakeNeural(newFakeStream(480, 480)), AttenuationLimitDB: 30, ToneMatch: 100, CeilingDB: -1}
	l, err := NewLive(opt, 0)
	if err != nil {
		t.Fatal(err)
	}
	base := l.Latency()
	l.SetBlockSize(480)
	out := make([]float32, len(x))
	for s := 0; s+480 <= len(x); s += 480 {
		l.Process(x[s:s+480], out[s:s+480])
	}
	if l.Latency() != base-480 {
		t.Errorf("tone matching changed the latency: %d vs %d", l.Latency(), base-480)
	}
	if l.fir.identity {
		t.Fatal("the tone filter was never learned")
	}
	// The thin voice should have gained bass relative to the mids.
	active := make([]bool, len(x))
	for i := range active {
		active[i] = math.Mod(float64(i)/testRate, 2.5) < 1.6
	}
	d := l.Latency()
	ref := make([]float32, len(x)-d)
	for i := range ref {
		ref[i] = 0.5 * x[i] // what the fake alone would give
	}
	from := 20 * testRate
	got := out[d:]
	bassGain := bandRMS(got[from:len(ref)], 100, 250, active[from:len(ref)], true) - bandRMS(ref[from:], 100, 250, active[from:len(ref)], true)
	midGain := bandRMS(got[from:len(ref)], 600, 1500, active[from:len(ref)], true) - bandRMS(ref[from:], 600, 1500, active[from:len(ref)], true)
	t.Logf("after learning: bass %+.1f dB, mids %+.1f dB", bassGain, midGain)
	if bassGain-midGain < 5 {
		t.Errorf("bass only %.1f dB above the mids", bassGain-midGain)
	}
}
