package deepfilter

import (
	"math"
	"os"
	"testing"
	"time"
)

// These tests run against the real DeepFilterNet; they are skipped unless
// KYONE_DF_LIB / KYONE_DF_MODEL / KYONE_DF_BIN point at it.

func noisyVoice(seconds, sigma float64) (noisy, clean []float32) {
	return SyntheticVoice(seconds, sigma)
}

func rms(x []float32, from, to int) float64 {
	var s float64
	for _, v := range x[from:to] {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(to-from))
}

func TestRealLibStream(t *testing.T) {
	libPath, model := os.Getenv("KYONE_DF_LIB"), os.Getenv("KYONE_DF_MODEL")
	if libPath == "" || model == "" {
		t.Skip("set KYONE_DF_LIB and KYONE_DF_MODEL to run against the real library")
	}
	lib, err := OpenLib(libPath)
	if err != nil {
		t.Fatal(err)
	}
	s, err := lib.NewStream(model, 100, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.FrameLength() != 480 {
		t.Fatalf("frame length %d, want 480 (10 ms)", s.FrameLength())
	}

	noisy, clean := noisyVoice(12, 0.01)
	out := make([]float32, len(noisy))
	f := s.FrameLength()
	start := time.Now()
	for i := 0; i+f <= len(noisy); i += f {
		s.Process(noisy[i:i+f], out[i:i+f])
	}
	rtf := time.Since(start).Seconds() / 12
	t.Logf("real-time factor %.3f (%.1f%% of one core)", rtf, 100*rtf)
	t.Logf("Latency() reports %d samples", s.Latency())

	// Latency: the delay that best aligns the output with the clean voice.
	best, bestV := 0, math.Inf(-1)
	for lag := 0; lag < 3000; lag++ {
		var v float64
		for i := 4 * SampleRate; i < 8*SampleRate; i += 2 {
			v += float64(out[i]) * float64(clean[i-lag])
		}
		if v > bestV {
			best, bestV = lag, v
		}
	}
	t.Logf("measured delay %d samples (%.1f ms)", best, 1000*float64(best)/SampleRate)

	// Noise in a pause (2.1–2.4 s into each 2.5 s period) before and after.
	var before, after float64
	for p := 0; p < 4; p++ {
		a, b := int((float64(p)*2.5+1.9)*SampleRate), int((float64(p)*2.5+2.4)*SampleRate)
		before += rms(noisy, a, b)
		after += rms(out, a+best, b+best)
	}
	t.Logf("pause level: %.1f dBFS -> %.1f dBFS", 20*math.Log10(before/4), 20*math.Log10(after/4))
	if after > before*0.6 {
		t.Errorf("DeepFilterNet barely reduced the noise (%.4f -> %.4f)", before/4, after/4)
	}
}

func TestRealCLI(t *testing.T) {
	bin := os.Getenv("KYONE_DF_BIN")
	if bin == "" {
		t.Skip("set KYONE_DF_BIN to run against the real deep-filter")
	}
	noisy, _ := noisyVoice(8, 0.01)
	cli := &CLI{Bin: bin}
	out, err := cli.Denoise(noisy, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(noisy) {
		t.Fatalf("length %d, want %d", len(out), len(noisy))
	}
	a, b := int(1.9*SampleRate), int(2.4*SampleRate)
	t.Logf("pause level: %.4f -> %.4f", rms(noisy, a, b), rms(out, a, b))
	if rms(out, a, b) > rms(noisy, a, b)*0.6 {
		t.Error("the CLI barely reduced the noise")
	}
	same, err := cli.Denoise(noisy, 0)
	if err != nil || len(same) != len(noisy) || same[1000] != noisy[1000] {
		t.Errorf("0 dB must return the input unchanged (%v)", err)
	}
}
