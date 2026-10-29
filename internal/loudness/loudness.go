// Package loudness implements ITU-R BS.1770-4 / EBU R128 measurements and the
// simple level statistics used to describe a voice recording.
package loudness

import (
	"math"
	"sort"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// Result summarises a measurement. Values the signal cannot support
// (e.g. loudness of digital silence) are reported as -Inf.
type Result struct {
	IntegratedLUFS float64 // gated programme loudness
	LRA            float64 // loudness range in LU
	TruePeakDB     float64 // dBTP, 4x oversampled
	SamplePeakDB   float64 // dBFS
	RMSDB          float64 // dBFS, whole file, all channels
	NoiseFloorDB   float64 // dBFS, 10th percentile of 50 ms RMS
	SpeechLevelDB  float64 // dBFS, 90th percentile of 50 ms RMS
	Clipped        int     // samples at or beyond full scale
}

// SNR is the spread between the speech level and the noise floor, a rough
// estimate of the recording's signal-to-noise ratio.
func (r Result) SNR() float64 { return r.SpeechLevelDB - r.NoiseFloorDB }

// kWeighting returns the two-stage BS.1770 pre-filter for any sample rate.
func kWeighting(rate float64) dsp.Cascade {
	// Stage 1: high shelf modelling the head.
	const f1, g1, q1 = 1681.974450955533, 3.999843853973347, 0.7071752369554196
	k := math.Tan(math.Pi * f1 / rate)
	vh := math.Pow(10, g1/20)
	vb := math.Pow(vh, 0.4996667741545416)
	a0 := 1 + k/q1 + k*k
	shelf := dsp.NewBiquad(
		(vh+vb*k/q1+k*k)/a0, 2*(k*k-vh)/a0, (vh-vb*k/q1+k*k)/a0,
		1, 2*(k*k-1)/a0, (1-k/q1+k*k)/a0)

	// Stage 2: RLB high-pass.
	const f2, q2 = 38.13547087602444, 0.5003270373238773
	k = math.Tan(math.Pi * f2 / rate)
	a0 = 1 + k/q2 + k*k
	hp := dsp.NewBiquad(1, -2, 1, 1, 2*(k*k-1)/a0, (1-k/q2+k*k)/a0)
	return dsp.Cascade{shelf, hp}
}

// Measure analyses b. It works at any sample rate; BS.1770 filters are
// derived for the buffer's rate.
func Measure(b *audio.Buffer) Result {
	r := Result{
		IntegratedLUFS: math.Inf(-1), LRA: 0,
		TruePeakDB: math.Inf(-1), SamplePeakDB: math.Inf(-1),
		RMSDB: math.Inf(-1), NoiseFloorDB: math.Inf(-1), SpeechLevelDB: math.Inf(-1),
	}
	if b.Frames() == 0 {
		return r
	}

	var peak, sumSq float64
	for _, ch := range b.Ch {
		for _, v := range ch {
			a := math.Abs(float64(v))
			if a > peak {
				peak = a
			}
			if a >= 0.99997 {
				r.Clipped++
			}
			sumSq += a * a
		}
	}
	r.SamplePeakDB = db(peak)
	r.RMSDB = db(math.Sqrt(sumSq / float64(b.Frames()*b.Channels())))
	var tp float64
	for _, ch := range b.Ch {
		tp = math.Max(tp, dsp.TruePeak(ch))
	}
	r.TruePeakDB = db(tp)

	r.IntegratedLUFS, r.LRA = gatedLoudness(b)
	r.NoiseFloorDB, r.SpeechLevelDB = frameLevels(b)
	return r
}

// gatedLoudness returns integrated loudness and loudness range.
func gatedLoudness(b *audio.Buffer) (lufs, lra float64) {
	rate := float64(b.Rate)
	step := int(math.Round(rate * 0.1)) // 100 ms sub-blocks
	nSteps := b.Frames() / step
	if nSteps < 4 {
		return math.Inf(-1), 0
	}
	// Mean-square energy of each sub-block, summed over channels (weight 1.0).
	sub := make([]float64, nSteps)
	for _, ch := range b.Ch {
		f := kWeighting(rate)
		for i := 0; i < nSteps; i++ {
			var s float64
			for _, v := range ch[i*step : (i+1)*step] {
				y := f.Process(float64(v))
				s += y * y
			}
			sub[i] += s / float64(step)
		}
	}

	toLUFS := func(z float64) float64 { return -0.691 + 10*math.Log10(z) }
	window := func(n int) []float64 { // sliding mean of n sub-blocks
		out := make([]float64, 0, nSteps-n+1)
		var acc float64
		for i, z := range sub {
			acc += z
			if i >= n {
				acc -= sub[i-n]
			}
			if i >= n-1 {
				out = append(out, acc/float64(n))
			}
		}
		return out
	}
	gate := func(blocks []float64, rel float64) (kept []float64, mean float64) {
		var sum float64
		var n int
		for _, z := range blocks {
			if z > 0 && toLUFS(z) > -70 {
				sum += z
				n++
			}
		}
		if n == 0 {
			return nil, 0
		}
		thr := toLUFS(sum/float64(n)) + rel
		sum, n = 0, 0
		for _, z := range blocks {
			if z > 0 && toLUFS(z) > thr {
				kept = append(kept, z)
				sum += z
				n++
			}
		}
		if n == 0 {
			return nil, 0
		}
		return kept, sum / float64(n)
	}

	// Integrated loudness: 400 ms blocks, relative gate at -10 LU.
	if kept, mean := gate(window(4), -10); kept != nil {
		lufs = toLUFS(mean)
	} else {
		lufs = math.Inf(-1)
	}

	// Loudness range: 3 s blocks, relative gate at -20 LU, L95 - L10.
	if nSteps >= 30 {
		if kept, _ := gate(window(30), -20); len(kept) > 1 {
			l := make([]float64, len(kept))
			for i, z := range kept {
				l[i] = toLUFS(z)
			}
			sort.Float64s(l)
			lra = percentile(l, 0.95) - percentile(l, 0.10)
		}
	}
	return lufs, lra
}

// Levels returns the estimated noise floor and active speech level in dBFS.
func Levels(b *audio.Buffer) (noise, speech float64) { return frameLevels(b) }

// frameLevels estimates the noise floor and the active speech level from the
// distribution of 50 ms RMS values of the channel mix.
func frameLevels(b *audio.Buffer) (noise, speech float64) {
	win := int(float64(b.Rate) * 0.05)
	n := b.Frames() / win
	if n < 10 {
		return math.Inf(-1), math.Inf(-1)
	}
	levels := make([]float64, n)
	inv := 1 / float64(b.Channels())
	for i := range levels {
		var s float64
		for j := i * win; j < (i+1)*win; j++ {
			var m float64
			for _, ch := range b.Ch {
				m += float64(ch[j])
			}
			m *= inv
			s += m * m
		}
		levels[i] = db(math.Sqrt(s / float64(win)))
	}
	sort.Float64s(levels)
	return percentile(levels, 0.10), percentile(levels, 0.90)
}

// db is dsp.DB that reports digital silence as -Inf instead of a -200 dB floor.
func db(x float64) float64 {
	if x <= 0 {
		return math.Inf(-1)
	}
	return dsp.DB(x)
}

// percentile interpolates the p-quantile (0..1) of an ascending slice.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	pos := p * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	f := pos - float64(lo)
	return sorted[lo]*(1-f) + sorted[hi]*f
}

// IntegratedLUFS is a convenience wrapper returning only the programme loudness.
func IntegratedLUFS(b *audio.Buffer) float64 {
	l, _ := gatedLoudness(b)
	return l
}
