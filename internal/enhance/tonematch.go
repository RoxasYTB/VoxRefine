package enhance

import (
	"math"
	"sort"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// studioTarget is the long-term spectrum of speech that tone matching steers
// towards: third-octave levels, in dB, of an Adobe Podcast Enhance rendering
// of a laptop-microphone recording, measured over its speech frames with
// toneAnalysis. Only its broad shape is used (it is smoothed before use), so
// the voice-specific detail (pitch peak, formants) cancels out of the
// comparison with the input; what remains is a warm low end, a gentle slope
// through the mids and a lifted 5 kHz presence region.
var studioTarget = [...]float64{
	-49.4, -38.5, -42.8, -46.6, -41.4, -41.4, -42.0, -47.6, // 100 … 504 Hz
	-48.4, -53.2, -55.1, -56.0, -54.4, -55.8, -56.5, -55.2, // 635 … 3.2 kHz
	-52.5, -47.1, -49.8, -52.4, -54.5, -65.9, -75.9, // 4 … 16 kHz
}

const (
	toneFFT      = 2048
	toneMinBoost = -12.0 // dB, deepest automatic cut
	toneMinSNR   = 12.0  // dB speech-over-noise a band needs to be boosted fully
	toneEmptyDB  = 40.0  // dB below the loudest band: the voice has no content there
	toneMinHz    = 250   // level reference band for the overall offset
	toneMaxHz    = 3200
	toneMinFrame = 20 // fewer speech frames than this: leave the tone alone
)

// toneCentres are the third-octave band centres, 100 Hz up, shared with the
// target table: 100·2^(i/3).
var toneCentres = func() []float64 {
	var c []float64
	for i := 0; ; i++ {
		f := 100 * math.Pow(2, float64(i)/3)
		if f >= 20000 {
			return c
		}
		c = append(c, f)
	}
}()

// toneAnalysis returns the third-octave speech level and noise level of b in
// dB (same scale as studioTarget). Speech frames are the loudest 45 % of the
// 50 ms blocks, noise frames the quietest 15 %. ok is false when there is too
// little speech to judge.
func toneAnalysis(b *audio.Buffer) (speech, noise []float64, ok bool) {
	rate := float64(b.Rate)
	n := b.Frames()
	mix := make([]float32, n)
	inv := 1 / float32(b.Channels())
	for _, ch := range b.Ch {
		for i, v := range ch {
			mix[i] += v * inv
		}
	}

	win := int(0.05 * rate)
	blocks := n / win
	if blocks < 20 {
		return nil, nil, false
	}
	level := make([]float64, blocks)
	for i := range level {
		var e float64
		for _, v := range mix[i*win : (i+1)*win] {
			e += float64(v) * float64(v)
		}
		level[i] = e / float64(win)
	}
	sorted := append([]float64(nil), level...)
	sort.Float64s(sorted)
	lowThr, highThr := sorted[int(0.15*float64(blocks))], sorted[int(0.55*float64(blocks))]
	inSpeech := func(centre int) bool { return level[min(centre/win, blocks-1)] >= highThr }
	inNoise := func(centre int) bool { return level[min(centre/win, blocks-1)] <= lowThr }

	fft := dsp.NewRFFT(toneFFT)
	buf := make([]float64, toneFFT)
	spec := make([]complex128, fft.Bins())
	accS := make([]float64, fft.Bins())
	accN := make([]float64, fft.Bins())
	var nS, nN int
	for s := 0; s+toneFFT <= n; s += toneFFT / 2 {
		centre := s + toneFFT/2
		isS, isN := inSpeech(centre), inNoise(centre)
		if !isS && !isN {
			continue
		}
		for i := range buf {
			buf[i] = float64(mix[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/toneFFT))
		}
		fft.Forward(buf, spec)
		for k, v := range spec {
			p := real(v)*real(v) + imag(v)*imag(v)
			if isS {
				accS[k] += p
			}
			if isN {
				accN[k] += p
			}
		}
		if isS {
			nS++
		}
		if isN {
			nN++
		}
	}
	if nS < toneMinFrame || nN == 0 {
		return nil, nil, false
	}
	return bandDB(accS, nS, rate), bandDB(accN, nN, rate), true
}

// bandDB sums the mean power spectrum acc/frames into third-octave bands.
func bandDB(acc []float64, frames int, rate float64) []float64 {
	out := make([]float64, len(toneCentres))
	for b, fc := range toneCentres {
		lo, hi := fc/math.Pow(2, 1.0/6), fc*math.Pow(2, 1.0/6)
		var e float64
		for k, p := range acc {
			if f := float64(k) * rate / toneFFT; f >= lo && f < hi {
				e += p
			}
		}
		out[b] = 10 * math.Log10(e/float64(frames)/toneFFT/toneFFT+1e-20)
	}
	return out
}

// smooth3 applies [1 2 1]/4 across bands, twice (≈ ±⅔ octave).
func smooth3(x []float64) []float64 {
	cur := append([]float64(nil), x...)
	for pass := 0; pass < 2; pass++ {
		next := make([]float64, len(cur))
		for i := range cur {
			l, r := cur[max(i-1, 0)], cur[min(i+1, len(cur)-1)]
			next[i] = 0.25*l + 0.5*cur[i] + 0.25*r
		}
		cur = next
	}
	return cur
}

// maxBoost is the largest automatic boost at a band centre: a warm low end
// may be restored generously, presence only moderately, and air barely, since
// lifting the top octaves mostly amplifies artefacts.
func maxBoost(hz float64) float64 {
	switch {
	case hz <= 2000:
		return 15
	case hz <= 8000:
		return 9
	default:
		return 4
	}
}

// toneCorrection returns the per-band gain, in dB, that moves the speech
// spectrum of the signal (shape measured after denoising) towards studioTarget
// at the given strength (0..1). It is level-neutral (zero on average over
// 250 Hz–3.2 kHz) and bounded, and it never boosts a band the voice does not
// really occupy: one whose level sits far below the loudest band, or that
// stood less than ~12 dB above the noise in the recording before denoising
// (rawSpeech and rawNoise, measured then).
func toneCorrection(speech, rawSpeech, rawNoise []float64, strength float64) []float64 {
	target := smooth3(studioTarget[:])
	measured := smooth3(speech)

	var sum, w float64
	for b, fc := range toneCentres {
		if fc >= toneMinHz && fc <= toneMaxHz {
			sum += target[b] - measured[b]
			w++
		}
	}
	offset := sum / w

	peak := math.Inf(-1)
	for _, v := range speech {
		peak = math.Max(peak, v)
	}

	corr := make([]float64, len(toneCentres))
	for b, fc := range toneCentres {
		c := target[b] - measured[b] - offset
		c = math.Min(math.Max(c, toneMinBoost), maxBoost(fc))
		if c > 0 {
			if speech[b] < peak-toneEmptyDB {
				c = 0
			}
			snr := rawSpeech[b] - rawNoise[b]
			c *= math.Min(math.Max(snr/toneMinSNR, 0), 1)
		}
		corr[b] = c
	}
	corr = smoothOnce(corr)
	for b := range corr {
		corr[b] *= strength
	}
	return corr
}

// smoothOnce applies [1 2 1]/4 across bands.
func smoothOnce(x []float64) []float64 {
	out := make([]float64, len(x))
	for i := range x {
		out[i] = 0.25*x[max(i-1, 0)] + 0.5*x[i] + 0.25*x[min(i+1, len(x)-1)]
	}
	return out
}

// toneGains expands the band corrections into a per-bin linear gain curve for
// the spectral engine, interpolating in dB over log frequency and tapering to
// zero below 40 Hz and above the Nyquist side of the last band.
func toneGains(corr []float64, rate float64) []float64 {
	gains := make([]float64, nBins)
	for k := range gains {
		f := float64(k) * rate / fftSize
		var db float64
		switch {
		case f <= 40:
			db = 0
		case f < toneCentres[0]:
			t := math.Log(f/40) / math.Log(toneCentres[0]/40)
			db = corr[0] * t
		case f >= toneCentres[len(toneCentres)-1]:
			last := toneCentres[len(toneCentres)-1]
			t := math.Max(1-math.Log(f/last)/math.Log(rate/2/last), 0)
			db = corr[len(corr)-1] * t
		default:
			i := sort.SearchFloat64s(toneCentres, f) // first centre >= f
			lo, hi := toneCentres[i-1], toneCentres[i]
			t := math.Log(f/lo) / math.Log(hi/lo)
			db = corr[i-1]*(1-t) + corr[i]*t
		}
		gains[k] = dsp.FromDB(db)
	}
	return gains
}

// toneProbe is what the tone matcher needs to know about the recording before
// it is denoised.
type toneProbe struct{ speech, noise []float64 }

// probeTone measures b's speech and noise spectra; nil if there is too little
// speech to judge.
func probeTone(b *audio.Buffer) *toneProbe {
	speech, noise, ok := toneAnalysis(b)
	if !ok {
		return nil
	}
	return &toneProbe{speech, noise}
}

// toneMatch steers the tonal balance of b (already denoised) towards the studio
// target. strength is 0..1. raw is the probe taken before denoising. It
// returns the applied per-band corrections in dB (nil when the stage was
// skipped for lack of speech).
func toneMatch(b *audio.Buffer, raw *toneProbe, strength float64) []float64 {
	if strength <= 0 || raw == nil || b.Frames() == 0 {
		return nil
	}
	speech, _, ok := toneAnalysis(b)
	if !ok {
		return nil
	}
	corr := toneCorrection(speech, raw.speech, raw.noise, strength)
	gains := toneGains(corr, float64(b.Rate))
	process(b, nil, func(pos int, pow, noise []float64) []float64 { return gains })
	return corr
}
