package loudness

import (
	"math"
	"math/rand"
	"testing"

	"kyone/internal/audio"
)

func tone(rate int, freq, amp, seconds float64) *audio.Buffer {
	n := int(float64(rate) * seconds)
	b := audio.NewBuffer(rate, 1, n)
	for i := range b.Ch[0] {
		b.Ch[0][i] = float32(amp * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return b
}

func TestIntegratedLoudnessOfReferenceTone(t *testing.T) {
	// BS.1770: a 997 Hz sine of amplitude A on one channel reads
	// 20·log10(A) - 3.01 LUFS.
	for _, rate := range []int{48000, 44100} {
		for _, amp := range []float64{0.1, 0.5} {
			got := Measure(tone(rate, 997, amp, 5)).IntegratedLUFS
			want := 20*math.Log10(amp) - 3.01
			if math.Abs(got-want) > 0.1 {
				t.Errorf("%d Hz amp %.1f: got %.2f LUFS want %.2f", rate, amp, got, want)
			}
		}
	}
}

func TestStereoSumsChannelEnergy(t *testing.T) {
	mono := tone(48000, 997, 0.1, 5)
	st := mono.ToChannels(2)
	d := Measure(st).IntegratedLUFS - Measure(mono).IntegratedLUFS
	if math.Abs(d-3.01) > 0.05 {
		t.Fatalf("stereo should read +3 LU, got %+.2f", d)
	}
}

func TestGatingIgnoresSilence(t *testing.T) {
	loud := tone(48000, 997, 0.1, 20)
	padded := audio.NewBuffer(48000, 1, 48000*60)
	copy(padded.Ch[0][48000*20:], loud.Ch[0])
	a, b := Measure(loud).IntegratedLUFS, Measure(padded).IntegratedLUFS
	// Without gating the padded file would read about 4.8 LU quieter.
	if math.Abs(a-b) > 0.15 {
		t.Fatalf("silence changed loudness: %.2f vs %.2f", a, b)
	}
}

func TestTruePeakExceedsSamplePeakForInterSamplePeaks(t *testing.T) {
	// A tone at fs/4 with a 45° phase offset samples at ±0.707 of its peak.
	n := 4800
	b := audio.NewBuffer(48000, 1, n)
	for i := range b.Ch[0] {
		b.Ch[0][i] = float32(math.Sin(2*math.Pi*float64(i)/4 + math.Pi/4))
	}
	r := Measure(b)
	if r.SamplePeakDB > -2.9 || r.SamplePeakDB < -3.2 {
		t.Fatalf("sample peak %.2f dBFS, want ≈ -3.01", r.SamplePeakDB)
	}
	if r.TruePeakDB < -0.3 {
		t.Fatalf("true peak %.2f dBTP, want ≈ 0", r.TruePeakDB)
	}
}

func TestLevelsOfNoiseAndBursts(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	rate := 48000
	b := audio.NewBuffer(rate, 1, rate*20)
	for i := range b.Ch[0] {
		amp := 0.003 // floor ≈ -50 dBFS
		if (i/rate)%2 == 0 {
			amp = 0.1 // speech-like bursts every other second
		}
		b.Ch[0][i] = float32(amp * rng.NormFloat64())
	}
	r := Measure(b)
	if math.Abs(r.NoiseFloorDB+50.5) > 1.5 {
		t.Errorf("noise floor %.1f dBFS, want ≈ -50", r.NoiseFloorDB)
	}
	if math.Abs(r.SpeechLevelDB+20) > 1.5 {
		t.Errorf("speech level %.1f dBFS, want ≈ -20", r.SpeechLevelDB)
	}
	if snr := r.SNR(); math.Abs(snr-30) > 3 {
		t.Errorf("SNR %.1f dB, want ≈ 30", snr)
	}
}

func TestSilenceIsMinusInf(t *testing.T) {
	r := Measure(audio.NewBuffer(48000, 1, 48000))
	if !math.IsInf(r.IntegratedLUFS, -1) || !math.IsInf(r.SamplePeakDB, -1) {
		t.Fatalf("got %+v", r)
	}
}
