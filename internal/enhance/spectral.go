package enhance

import (
	"math"
	"runtime"
	"slices"
	"sort"
	"sync"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// Analysis/synthesis geometry. Frames are 1024 samples (21 ms at 48 kHz) with
// a 75 % overlap and a sine window, which satisfies the COLA condition for
// weighted overlap-add (Σ w² = 2).
const (
	fftSize = 1024
	hopSize = 256
	nBins   = fftSize/2 + 1

	// The noise tracker looks at frames every strideSize samples and publishes
	// one noise spectrum every gridEvery strides, using a window of
	// ±noiseHalfWindow strides (≈ ±2 s at 48 kHz).
	strideSize      = 512
	quietFraction   = 0.10 // share of frames taken as "noise only"
	gridEvery       = 8
	noiseHalfWindow = 188

	// Band used to decide whether a frame is quiet (≈ 190 Hz – 4.5 kHz).
	quietLoBin = 4
	quietHiBin = 96

	lambdaFloor = 1e-12 // keeps divisions finite on digital silence
)

var window = func() []float64 {
	w := make([]float64, fftSize)
	for i := range w {
		w[i] = math.Sin(math.Pi * float64(i) / fftSize)
	}
	return w
}()

// SpectralConfig drives the denoise / dereverberation stage.
type SpectralConfig struct {
	// AttenuationDB caps how far any time-frequency cell is turned down. Zero
	// disables noise reduction altogether.
	AttenuationDB float64
	// Overshoot (>= 1) scales the tracked noise spectrum; higher values remove
	// more noise at the price of more speech colouration.
	Overshoot float64
	// SpeechFloorDB is the shallower attenuation limit used while speech is
	// present (voice protection): between words the full AttenuationDB applies,
	// but inside speech weak detail (breath, soft consonants, inter-harmonic
	// energy) is only turned down this far, which avoids holes and a "watery"
	// sound. Zero or a value above AttenuationDB means no voice protection.
	SpeechFloorDB float64
	// Dereverb is the late-reverberation suppression strength, 0..1.
	Dereverb float64
	// ReverbTime is the assumed RT60 in seconds; zero selects 0.45 s.
	ReverbTime float64
}

// Active reports whether the stage would change the signal.
func (c SpectralConfig) Active() bool { return c.AttenuationDB > 0 || c.Dereverb > 0 }

// Spectral applies noise reduction and late-reverb suppression to every
// channel of b. The gain is computed from the channel-averaged power so the
// stereo image is preserved. b is modified in place.
func Spectral(b *audio.Buffer, cfg SpectralConfig) {
	if !cfg.Active() || b.Frames() == 0 {
		return
	}
	if cfg.Overshoot < 1 {
		cfg.Overshoot = 1
	}
	if cfg.ReverbTime <= 0 {
		cfg.ReverbTime = 0.45
	}
	var nm *noiseMap
	if cfg.AttenuationDB > 0 {
		nm = estimateNoise(b)
	}
	gs := newGainState(cfg, b.Rate)
	gs.clicks = nm
	process(b, nm, gs.update)
}

// noiseMap holds the tracked noise power spectrum on a coarse time grid.
type noiseMap struct {
	spectra [][]float32 // [grid][nBins]
	spacing int         // samples between grid points
	clicks  [][2]int    // sample ranges of isolated clicks, ascending
}

// inClick reports whether sample position pos lies inside an isolated click.
func (m *noiseMap) inClick(pos int) bool {
	if m == nil {
		return false
	}
	i := sort.Search(len(m.clicks), func(i int) bool { return m.clicks[i][1] >= pos })
	return i < len(m.clicks) && m.clicks[i][0] <= pos
}

// at writes the noise spectrum interpolated at sample position pos into dst.
func (m *noiseMap) at(pos int, dst []float64) {
	x := float64(pos) / float64(m.spacing)
	last := len(m.spectra) - 1
	if x <= 0 {
		for k, v := range m.spectra[0] {
			dst[k] = float64(v)
		}
		return
	}
	if x >= float64(last) {
		for k, v := range m.spectra[last] {
			dst[k] = float64(v)
		}
		return
	}
	i := int(x)
	f := x - float64(i)
	a, b := m.spectra[i], m.spectra[i+1]
	for k := range dst {
		dst[k] = float64(a[k])*(1-f) + float64(b[k])*f
	}
}

// analyzer computes windowed power spectra; it owns scratch memory, so each
// goroutine needs its own.
type analyzer struct {
	fft  *dsp.RFFT
	buf  []float64
	spec []complex128
}

func newAnalyzer() *analyzer {
	return &analyzer{fft: dsp.NewRFFT(fftSize), buf: make([]float64, fftSize), spec: make([]complex128, nBins)}
}

func loadFrame(dst []float64, x []float32, start int) {
	for i := range dst {
		j := start + i
		if j >= 0 && j < len(x) {
			dst[i] = float64(x[j]) * window[i]
		} else {
			dst[i] = 0
		}
	}
}

// power fills pow with the channel-averaged power spectrum of the frame
// starting at sample index start (which may be negative or run past the end).
func (a *analyzer) power(b *audio.Buffer, start int, pow []float64) {
	for k := range pow {
		pow[k] = 0
	}
	for _, ch := range b.Ch {
		loadFrame(a.buf, ch, start)
		a.fft.Forward(a.buf, a.spec)
		for k, v := range a.spec {
			pow[k] += real(v)*real(v) + imag(v)*imag(v)
		}
	}
	inv := 1 / float64(b.Channels())
	for k := range pow {
		pow[k] *= inv
	}
}

// estimateNoise tracks the background noise spectrum over time.
//
// For every grid point it looks at the surrounding ±2 s, keeps the quietest
// 15 % of frames (judged on speech-band energy) and averages their power
// spectra. Because the whole file is available, the window is centred on the
// grid point, which lets the estimate follow slowly changing noise without the
// lag and bias of causal minimum-statistics trackers.
func estimateNoise(b *audio.Buffer) *noiseMap {
	nFrames := b.Frames()/strideSize + 1
	nGrid := (nFrames-1)/gridEvery + 1

	// Pass 1: broadband speech-band energy of every stride frame. The frames
	// are independent, so they are spread across all cores.
	energy := make([]float32, nFrames)
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			an := newAnalyzer()
			pow := make([]float64, nBins)
			for f := w; f < nFrames; f += workers {
				an.power(b, f*strideSize-fftSize/2, pow)
				var e float64
				for _, p := range pow[quietLoBin : quietHiBin+1] {
					e += p
				}
				energy[f] = float32(e)
			}
		}(w)
	}
	wg.Wait()

	// Per-grid-point energy threshold selecting the quiet frames.
	thr := make([]float32, nGrid)
	scratch := make([]float32, 0, 2*noiseHalfWindow+1)
	for g := range thr {
		lo, hi := windowBounds(g, nFrames)
		scratch = append(scratch[:0], energy[lo:hi+1]...)
		slices.Sort(scratch)
		idx := int(quietFraction * float64(len(scratch)))
		if idx >= len(scratch) {
			idx = len(scratch) - 1
		}
		thr[g] = scratch[idx]
	}

	clicks := findClicks(energy, thr)

	// Pass 2: accumulate the spectra of quiet frames into the grid points
	// whose window contains them, finalising each grid point as soon as the
	// last frame of its window has been seen.
	nm := &noiseMap{spectra: make([][]float32, nGrid), spacing: gridEvery * strideSize, clicks: clicks}
	type accum struct {
		sum []float64
		n   int
	}
	active := make(map[int]*accum)
	an := newAnalyzer()
	pow := make([]float64, nBins)
	finalise := func(g int) {
		a := active[g]
		delete(active, g)
		out := make([]float32, nBins)
		if a != nil && a.n > 0 {
			inv := 1 / float64(a.n)
			smoothed := smoothBins(a.sum, 2)
			for k, v := range smoothed {
				out[k] = float32(v * inv)
			}
		}
		nm.spectra[g] = out
	}
	for f := 0; f < nFrames; f++ {
		an.power(b, f*strideSize-fftSize/2, pow)
		gLo := ceilDiv(f-noiseHalfWindow, gridEvery)
		gHi := (f + noiseHalfWindow) / gridEvery
		if gLo < 0 {
			gLo = 0
		}
		if gHi > nGrid-1 {
			gHi = nGrid - 1
		}
		for g := gLo; g <= gHi; g++ {
			if energy[f] > thr[g] {
				continue
			}
			a := active[g]
			if a == nil {
				a = &accum{sum: make([]float64, nBins)}
				active[g] = a
			}
			for k, p := range pow {
				a.sum[k] += p
			}
			a.n++
		}
		// Grid points whose window ends at f are complete.
		if f >= noiseHalfWindow {
			if g := (f - noiseHalfWindow) / gridEvery; (f-noiseHalfWindow)%gridEvery == 0 && g < nGrid {
				finalise(g)
			}
		}
	}
	for g := 0; g < nGrid; g++ {
		if nm.spectra[g] == nil {
			finalise(g)
		}
	}
	return nm
}

// Click detection thresholds, in stride frames (≈ 10.7 ms each).
const (
	clickMaxFrames = 3   // longest burst still called a click (≈ 32 ms)
	clickNearGap   = 4   // quiet guard around the burst starts this far away…
	clickFarGap    = 12  // …and extends to this far (≈ 43–128 ms)
	clickRatio     = 3.0 // energy over the local noise level that counts as raised
)

// findClicks locates isolated short bursts: a few frames clearly above the
// local noise level, with nothing raised in the ~40–130 ms on either side.
// Words, even short ones, last longer and are followed or preceded by more
// voiced frames, so they are never mistaken for clicks; lip smacks, mouse
// clicks and similar ticks in the pauses are. thr is the per-grid-point
// noise level (the energy threshold of the quietest frames). The result is
// sample ranges, ascending.
func findClicks(energy, thr []float32) [][2]int {
	n := len(energy)
	raised := make([]bool, n)
	for f, e := range energy {
		g := min((f+gridEvery/2)/gridEvery, len(thr)-1)
		raised[f] = thr[g] > 0 && float64(e) > clickRatio*float64(thr[g])
	}
	anyRaised := func(lo, hi int) bool {
		for j := max(lo, 0); j <= min(hi, n-1); j++ {
			if raised[j] {
				return true
			}
		}
		return false
	}
	var out [][2]int
	for f := 0; f < n; {
		if !raised[f] {
			f++
			continue
		}
		a := f
		for f < n && raised[f] {
			f++
		}
		b := f - 1
		if b-a+1 > clickMaxFrames {
			continue
		}
		if anyRaised(a-clickFarGap, a-clickNearGap) || anyRaised(b+clickNearGap, b+clickFarGap) {
			continue
		}
		out = append(out, [2]int{(a-1)*strideSize - strideSize/2, (b+1)*strideSize + strideSize/2})
	}
	return out
}

func windowBounds(g, nFrames int) (lo, hi int) {
	lo = g*gridEvery - noiseHalfWindow
	hi = g*gridEvery + noiseHalfWindow
	if lo < 0 {
		lo = 0
	}
	if hi > nFrames-1 {
		hi = nFrames - 1
	}
	return
}

func ceilDiv(a, b int) int {
	if a >= 0 {
		return (a + b - 1) / b
	}
	return -((-a) / b)
}

// smoothBins returns the moving average of x over ±radius bins.
func smoothBins(x []float64, radius int) []float64 {
	out := make([]float64, len(x))
	for k := range x {
		lo, hi := k-radius, k+radius
		if lo < 0 {
			lo = 0
		}
		if hi >= len(x) {
			hi = len(x) - 1
		}
		var s float64
		for _, v := range x[lo : hi+1] {
			s += v
		}
		out[k] = s / float64(hi-lo+1)
	}
	return out
}

// gainState is the recursive state of the gain estimator.
type gainState struct {
	clicks       *noiseMap // supplies the isolated clicks to remove (may be nil)
	attenuation  float64   // linear gain floor between words
	softFloor    float64   // linear gain floor inside speech (>= attenuation)
	softFloorLow float64   // same below lowFloorBin, deeper
	speechProb   float64   // smoothed frame-level speech presence, 0..1
	overshoot    float64
	denoise      bool

	// Decision-directed a-priori SNR memory and speech-presence smoother.
	prevSNR []float64
	zeta    []float64

	// Late-reverberation model.
	useReverb bool
	rho       float64     // late-to-direct power ratio at full strength
	decay     float64     // per-hop power decay for the assumed RT60
	delay     int         // frames between direct sound and late reverb
	history   [][]float64 // ring of enhanced power spectra, one slot per delayed frame
	histPos   int
	reverb    []float64

	// Scratch.
	lam, xi, v, gl, xs, g []float64
}

const (
	ddAlpha       = 0.94 // decision-directed smoothing
	zetaSmooth    = 0.7  // temporal smoothing of the speech-presence SNR
	zetaLow       = 0.2  // below: speech surely absent
	zetaHigh      = 2.0  // above: speech surely present
	maxAbsentProb = 0.9

	// Frame-level speech detector (voice protection): band bins and the
	// a-posteriori SNR range mapped to presence 0..1, and its smoothing.
	lowFloorBin     = 7 // below ≈ 330 Hz the in-speech floor is deeper
	lowFloorExtraDB = 12.0
	speechLoBin     = 4  // ≈ 190 Hz
	speechHiBin     = 85 // ≈ 4 kHz
	speechOnsetDB   = 2.0
	speechFullDB    = 8.0
	speechAttack    = 0.3
	speechRelease   = 0.97
	reverbDelayMS   = 40.0
)

func newGainState(cfg SpectralConfig, rate int) *gainState {
	// Noise reduction and reverb suppression share one gain floor.
	floorDB := cfg.AttenuationDB
	if r := cfg.Dereverb * 15; r > floorDB {
		floorDB = r
	}
	s := &gainState{
		attenuation: dsp.FromDB(-floorDB),
		softFloor:   dsp.FromDB(-floorDB),
		overshoot:   cfg.Overshoot,
		denoise:     cfg.AttenuationDB > 0,
		prevSNR:     make([]float64, nBins),
		zeta:        make([]float64, nBins),
		reverb:      make([]float64, nBins),
		lam:         make([]float64, nBins),
		xi:          make([]float64, nBins),
		v:           make([]float64, nBins),
		gl:          make([]float64, nBins),
		xs:          make([]float64, nBins),
		g:           make([]float64, nBins),
	}
	s.softFloorLow = s.softFloor
	if cfg.SpeechFloorDB > 0 && cfg.SpeechFloorDB < floorDB {
		s.softFloor = dsp.FromDB(-cfg.SpeechFloorDB)
		s.softFloorLow = dsp.FromDB(-math.Min(cfg.SpeechFloorDB+lowFloorExtraDB, floorDB))
	}
	if cfg.Dereverb > 0 {
		s.useReverb = true
		s.rho = 0.35 * cfg.Dereverb
		hop := float64(hopSize) / float64(rate)
		s.decay = math.Pow(10, -6*hop/cfg.ReverbTime)
		s.delay = int(math.Round(reverbDelayMS / 1000 * float64(rate) / hopSize))
		if s.delay < 1 {
			s.delay = 1
		}
		s.history = make([][]float64, s.delay)
		for i := range s.history {
			s.history[i] = make([]float64, nBins)
		}
	}
	return s
}

// update computes the spectral gain for one frame given its power spectrum
// and the tracked noise spectrum, leaving the result in s.g.
func (s *gainState) update(pos int, pow, noise []float64) []float64 {
	// Late reverb: exponentially decaying memory of the enhanced power from
	// `delay` frames ago (Lebart/Habets-style statistical model).
	if s.useReverb {
		old := s.history[s.histPos] // oldest entry == enhanced power delay frames ago
		for k := range s.reverb {
			s.reverb[k] = s.decay*s.reverb[k] + (1-s.decay)*old[k]
		}
	}

	xiMin := s.attenuation * s.attenuation
	var bandPow, bandLam float64 // speech band, for the frame-level detector
	for k := 0; k < nBins; k++ {
		lam := 0.0
		if s.denoise {
			lam = noise[k] * s.overshoot
		}
		if s.useReverb {
			lam += s.rho * s.reverb[k]
		}
		if lam < lambdaFloor {
			lam = lambdaFloor
		}
		s.lam[k] = lam
		if k >= speechLoBin && k <= speechHiBin {
			bandPow += pow[k]
			bandLam += lam
		}

		gamma := pow[k] / lam
		xi := ddAlpha*s.prevSNR[k] + (1-ddAlpha)*math.Max(gamma-1, 0)
		if xi < xiMin {
			xi = xiMin
		}
		s.xi[k] = xi

		w := xi / (1 + xi)
		v := w * gamma
		if v < 1e-6 {
			v = 1e-6
		}
		s.v[k] = v
		gl := w // log-spectral-amplitude estimator; E1(v) vanishes for large v
		if v < 30 {
			gl *= math.Exp(0.5 * expint1(v))
		}
		if gl > 1 {
			gl = 1
		}
		s.gl[k] = gl
	}

	// Speech-presence probability from the locally smoothed a-priori SNR.
	for k := 0; k < nBins; k++ {
		lo, hi := k-3, k+3
		if lo < 0 {
			lo = 0
		}
		if hi >= nBins {
			hi = nBins - 1
		}
		var sum float64
		for _, x := range s.xi[lo : hi+1] {
			sum += x
		}
		s.xs[k] = sum / float64(hi-lo+1)
	}

	// Voice protection: the floor slides from the deep between-words limit to
	// the shallow in-speech limit as the frame's speech presence rises. The
	// detector is the band-wide a-posteriori SNR, smoothed with a fast attack
	// and a slow release so that word endings stay protected.
	snrDB := 10 * math.Log10(bandPow/bandLam+1e-12)
	target := math.Min(math.Max((snrDB-speechOnsetDB)/(speechFullDB-speechOnsetDB), 0), 1)
	coef := speechRelease
	if target > s.speechProb {
		coef = speechAttack
	}
	s.speechProb = coef*s.speechProb + (1-coef)*target
	logFloor := s.speechProb*math.Log(s.softFloor) + (1-s.speechProb)*math.Log(s.attenuation)
	floor := math.Exp(logFloor)
	// Below lowFloorBin the in-speech floor is deeper: speech there is strong
	// and harmonic, so what lies between harmonics is mostly rumble, and a
	// warm low end (tone matching boosts it) would otherwise lift that rumble.
	logFloorLow := s.speechProb*math.Log(s.softFloorLow) + (1-s.speechProb)*math.Log(s.attenuation)
	floorLow := math.Exp(logFloorLow)

	logSpan := math.Log(zetaHigh / zetaLow)
	for k := 0; k < nBins; k++ {
		s.zeta[k] = zetaSmooth*s.zeta[k] + (1-zetaSmooth)*s.xs[k]
		q := math.Log(zetaHigh/math.Max(s.zeta[k], 1e-9)) / logSpan
		q = math.Min(math.Max(q, 0), maxAbsentProb)
		// p = 1 / (1 + q/(1-q) · (1+ξ) · e^-v)
		odds := q / (1 - q) * (1 + s.xi[k]) * math.Exp(-math.Min(s.v[k], 50))
		p := 1 / (1 + odds)
		// OM-LSA: G = G_lsa^p · G_floor^(1-p)
		lf := logFloor
		if k < lowFloorBin {
			lf = logFloorLow
		}
		s.g[k] = math.Exp(p*math.Log(s.gl[k]) + (1-p)*lf)
	}

	// Light 3-tap smoothing across frequency, then the hard floor.
	prev := s.g[0]
	for k := 1; k < nBins-1; k++ {
		cur := s.g[k]
		s.g[k] = 0.25*prev + 0.5*cur + 0.25*s.g[k+1]
		prev = cur
	}
	for k := range s.g {
		fl := floor
		if k < lowFloorBin {
			fl = floorLow
		}
		if s.g[k] < fl {
			s.g[k] = fl
		}
		if s.g[k] > 1 {
			s.g[k] = 1
		}
	}

	// An isolated click is removed down to the deep floor, whatever its SNR.
	if s.clicks.inClick(pos) {
		for k := range s.g {
			s.g[k] = s.attenuation
		}
		s.speechProb = 0
	}

	// Feed the estimator's memories with this frame's outcome. The history
	// slot being overwritten held the enhanced power from `delay` frames ago,
	// which was consumed at the top of this call.
	for k := 0; k < nBins; k++ {
		clean := s.g[k] * s.g[k] * pow[k]
		s.prevSNR[k] = clean / s.lam[k]
		if s.useReverb {
			s.history[s.histPos][k] = clean
		}
	}
	if s.useReverb {
		s.histPos = (s.histPos + 1) % len(s.history)
	}
	return s.g
}

// process runs the frame loop: analysis FFT, gain estimation, synthesis FFT
// and weighted overlap-add.
func process(b *audio.Buffer, nm *noiseMap, gains func(pos int, pow, noise []float64) []float64) {
	n := b.Frames()
	nCh := b.Channels()
	fft := dsp.NewRFFT(fftSize)
	specs := make([][]complex128, nCh)
	for c := range specs {
		specs[c] = make([]complex128, nBins)
	}
	frame := make([]float64, fftSize)
	pow := make([]float64, nBins)
	noise := make([]float64, nBins)
	out := make([][]float32, nCh)
	for c := range out {
		out[c] = make([]float32, n)
	}

	const olaScale = 0.5 // Σ w² == 2 for a sine window at 75 % overlap
	invCh := 1 / float64(nCh)
	last := (n+fftSize)/hopSize + 1
	for t := -2; t <= last; t++ {
		start := t*hopSize - fftSize/2
		if start+fftSize <= 0 || start >= n {
			continue
		}
		for k := range pow {
			pow[k] = 0
		}
		for c := 0; c < nCh; c++ {
			loadFrame(frame, b.Ch[c], start)
			fft.Forward(frame, specs[c])
			for k, v := range specs[c] {
				pow[k] += real(v)*real(v) + imag(v)*imag(v)
			}
		}
		for k := range pow {
			pow[k] *= invCh
		}
		if nm != nil {
			nm.at(max(t*hopSize, 0), noise)
		}
		g := gains(max(t*hopSize, 0), pow, noise)

		for c := 0; c < nCh; c++ {
			for k := range specs[c] {
				specs[c][k] *= complex(g[k], 0)
			}
			fft.Inverse(specs[c], frame)
			dst := out[c]
			for i, v := range frame {
				if j := start + i; j >= 0 && j < n {
					dst[j] += float32(v * window[i] * olaScale)
				}
			}
		}
	}
	b.Ch = out
}

// expint1 is the exponential integral E1(x) for x > 0.
func expint1(x float64) float64 {
	const euler = 0.5772156649015329
	if x <= 1 {
		sum := -euler - math.Log(x)
		term := 1.0
		for k := 1; k <= 30; k++ {
			term *= -x / float64(k)
			d := term / float64(k)
			sum -= d
			if math.Abs(d) < 1e-14 {
				break
			}
		}
		return sum
	}
	// Continued fraction (modified Lentz).
	b := x + 1
	c := 1 / 1e-300
	d := 1 / b
	h := d
	for i := 1; i <= 100; i++ {
		a := -float64(i * i)
		b += 2
		d = 1 / (a*d + b)
		c = b + a/c
		del := c * d
		h *= del
		if math.Abs(del-1) < 1e-12 {
			break
		}
	}
	return h * math.Exp(-x)
}
