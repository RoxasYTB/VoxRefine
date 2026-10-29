package enhance

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"kyone/internal/audio"
	"kyone/internal/loudness"
)

// Report describes what Run did.
type Report struct {
	Before, After loudness.Result
	Stages        []StageTime
	GainDB        float64   // normalisation gain actually applied
	LimiterDB     float64   // deepest limiter gain reduction
	CompressorDB  float64   // deepest compressor gain reduction
	ToneMatchDB   []float64 // per-third-octave correction applied by tone matching, nil if skipped
	Resampled     bool
}

// StageTime is the wall-clock time one stage took.
type StageTime struct {
	Name     string
	Duration time.Duration
}

const (
	// maxNormalizeGainDB stops a near-silent file from being amplified into a
	// wall of noise.
	maxNormalizeGainDB = 40.0
	// maxLimiterDB bounds how hard the limiter may work to reach the target.
	maxLimiterDB      = 12.0
	loudnessTolerance = 0.05
)

// Run enhances in and returns the result at SampleRate. To avoid holding
// several copies of a long recording in memory, Run takes ownership of in:
// its samples may be overwritten and it must not be used afterwards.
// progress, if non-nil, is called with the name of each stage as it starts.
func Run(in *audio.Buffer, opt Options, progress func(stage string)) (*audio.Buffer, *Report, error) {
	if err := in.Validate(); err != nil {
		return nil, nil, err
	}
	if err := opt.Validate(); err != nil {
		return nil, nil, err
	}
	if in.Frames() < in.Rate/10 {
		return nil, nil, errors.New("recording is shorter than 100 ms")
	}

	rep := &Report{}
	stage := func(name string) func() {
		if progress != nil {
			progress(name)
		}
		t0 := time.Now()
		return func() { rep.Stages = append(rep.Stages, StageTime{name, time.Since(t0)}) }
	}

	done := stage("prepare")
	work := in
	if in.Rate != SampleRate {
		work = audio.Resample(in, SampleRate)
		rep.Resampled = true
	}
	if opt.Mono {
		work = work.ToChannels(1)
	}
	done()

	done = stage("analyse input")
	rep.Before = loudness.Measure(work)
	done()

	if opt.HighPassHz > 0 {
		done = stage("high-pass")
		highPass(work, opt.HighPassHz)
		done()
	}

	var toneRaw *toneProbe
	if opt.ToneMatch > 0 {
		toneRaw = probeTone(work) // before denoising: where is the voice above the noise?
	}

	cfg := SpectralConfig{
		AttenuationDB: opt.AttenuationLimitDB,
		Overshoot:     opt.Overshoot,
		SpeechFloorDB: opt.SpeechFloorDB,
		Dereverb:      opt.Dereverb / 100,
	}
	if opt.Neural != nil && opt.AttenuationLimitDB > 0 {
		done = stage("denoise (" + opt.Neural.Name + ")")
		if err := neuralDenoise(work, opt.Neural, opt.AttenuationLimitDB); err != nil {
			return nil, nil, err
		}
		done()
		// The neural denoiser has done the noise; kyone's own stage may still
		// shorten the room reverb.
		cfg.AttenuationDB, cfg.SpeechFloorDB = 0, 0
	}
	if cfg.Active() {
		done = stage("denoise / dereverb")
		Spectral(work, cfg)
		done()
	}

	if opt.ToneMatch > 0 {
		done = stage("tone match")
		rep.ToneMatchDB = toneMatch(work, toneRaw, opt.ToneMatch/100)
		done()
	}

	if opt.DeEss {
		done = stage("de-ess")
		deEss(work)
		done()
	}

	if opt.BassDB != 0 || opt.PresenceDB != 0 || opt.TrebleDB != 0 || len(opt.EQ) > 0 {
		done = stage("tone")
		shapeTone(work, opt)
		done()
	}

	if opt.Clarity > 0 {
		done = stage("clarity")
		noiseDB, speechDB := loudness.Levels(work)
		clarity(work, opt.Clarity/100, noiseDB, speechDB)
		done()
	}

	if opt.Compress {
		done = stage("level")
		// Place the threshold from the cleaned signal, not the noisy input.
		_, speechDB := loudness.Levels(work)
		rep.CompressorDB = compress(work, speechDB)
		done()
	}

	done = stage("loudness")
	if opt.Normalize {
		rep.GainDB, rep.LimiterDB = normalize(work, opt.TargetLUFS, opt.CeilingDB)
	} else {
		rep.LimiterDB = limit(work, opt.CeilingDB)
	}
	done()

	rep.After = loudness.Measure(work)
	if math.IsInf(rep.After.SamplePeakDB, -1) {
		return nil, nil, errors.New("the result is silent: the input contains no usable signal")
	}
	return work, rep, nil
}

// normalize scales b to targetLUFS and limits its true peak to ceilingDB. The
// limiter lowers the loudness a little, so the gain is refined iteratively:
// step towards the target, then bisect once a gain overshoots (loudness above
// target, or the limiter working harder than maxLimiterDB). It returns the
// gain applied and the limiter's deepest reduction. If the peaks do not allow
// the target, the loudest acceptable result is kept.
func normalize(b *audio.Buffer, targetLUFS, ceilingDB float64) (gainDB, limiterDB float64) {
	measured := loudness.IntegratedLUFS(b)
	if math.IsInf(measured, -1) || math.IsNaN(measured) {
		return 0, limit(b, ceilingDB)
	}
	base := b.Clone()
	apply := func(g float64) (lim, got float64) {
		scale := float32(math.Pow(10, g/20))
		for c := range b.Ch {
			for i, v := range base.Ch[c] {
				b.Ch[c][i] = v * scale
			}
		}
		lim = limit(b, ceilingDB)
		return lim, loudness.IntegratedLUFS(b)
	}

	g := math.Min(targetLUFS-measured, maxNormalizeGainDB)
	lo, hi := math.Inf(-1), math.Inf(1) // accepted / rejected gains
	var bestG, bestLim float64
	haveBest := false
	applied := math.NaN()
	for iter := 0; iter < 12; iter++ {
		lim, got := apply(g)
		applied = g
		ok := lim <= maxLimiterDB && got <= targetLUFS+loudnessTolerance
		if ok {
			lo, bestG, bestLim, haveBest = g, g, lim, true
			if got >= targetLUFS-loudnessTolerance || g >= maxNormalizeGainDB {
				break
			}
		} else {
			hi = g
		}
		var next float64
		switch {
		case !math.IsInf(lo, -1) && !math.IsInf(hi, 1):
			next = (lo + hi) / 2
		case ok:
			next = g + (targetLUFS - got)
		default:
			step := 3.0
			if got > targetLUFS+loudnessTolerance {
				step = got - targetLUFS
			}
			next = g - step
		}
		next = math.Min(next, maxNormalizeGainDB)
		if math.Abs(next-g) < 0.01 {
			break
		}
		g = next
	}
	if haveBest && applied != bestG {
		bestLim, _ = apply(bestG)
		applied = bestG
	}
	return applied, bestLim
}

// Timings renders the per-stage durations on one line.
func (r *Report) Timings() string {
	parts := make([]string, len(r.Stages))
	for i, st := range r.Stages {
		parts[i] = fmt.Sprintf("%s %.1fs", st.Name, st.Duration.Seconds())
	}
	return strings.Join(parts, ", ")
}

// ToneSummary describes the automatic tone correction in one line, or returns
// "" if none was applied.
func (r *Report) ToneSummary() string {
	if r.ToneMatchDB == nil {
		return ""
	}
	hi, lo := 0, 0
	for b, c := range r.ToneMatchDB {
		if c > r.ToneMatchDB[hi] {
			hi = b
		}
		if c < r.ToneMatchDB[lo] {
			lo = b
		}
	}
	out := ""
	if c := r.ToneMatchDB[hi]; c >= 1 {
		out += fmt.Sprintf("boost up to %+.1f dB around %.0f Hz", c, toneCentres[hi])
	}
	if c := r.ToneMatchDB[lo]; c <= -1 {
		if out != "" {
			out += ", "
		}
		out += fmt.Sprintf("cut up to %+.1f dB around %.0f Hz", c, toneCentres[lo])
	}
	if out == "" {
		out = "already close to the studio curve"
	}
	return out
}

// Summary renders a report as a few human-readable lines.
func (r *Report) Summary() string {
	f := func(v float64) string {
		if math.IsInf(v, -1) {
			return "  -inf"
		}
		return fmt.Sprintf("%6.1f", v)
	}
	return fmt.Sprintf(
		"                     before     after\n"+
			"  loudness (LUFS)  %s    %s\n"+
			"  true peak (dBTP) %s    %s\n"+
			"  noise floor (dB) %s    %s\n"+
			"  speech level (dB)%s    %s\n"+
			"  est. SNR (dB)    %s    %s\n"+
			"  loudness range   %s    %s",
		f(r.Before.IntegratedLUFS), f(r.After.IntegratedLUFS),
		f(r.Before.TruePeakDB), f(r.After.TruePeakDB),
		f(r.Before.NoiseFloorDB), f(r.After.NoiseFloorDB),
		f(r.Before.SpeechLevelDB), f(r.After.SpeechLevelDB),
		f(r.Before.SNR()), f(r.After.SNR()),
		f(r.Before.LRA), f(r.After.LRA),
	)
}
