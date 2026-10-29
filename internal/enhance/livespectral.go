package enhance

import (
	"math"
	"slices"

	"kyone/internal/dsp"
)

// streamSpectral is the causal form of the spectral stage: the same STFT
// geometry (1024-sample sine-window frames, hop 256) and the same OM-LSA gain
// estimator, fed by a noise tracker that only looks at the past and, if tone
// matching is on, by a slowly adapting studio-curve correction multiplied into
// the gain. Its delay is fftSize − hopSize samples (16 ms at 48 kHz).
type streamSpectral struct {
	fft   *dsp.RFFT
	in    []float64 // the latest fftSize input samples
	ola   []float64 // overlap-add accumulator
	frame []float64
	spec  []complex128
	pow   []float64
	gm    []float64 // final per-bin gain
	zeros []float64
	ones  []float64

	gs   *gainState // nil when only tone matching is on
	nt   *noiseTracker
	tone *liveTone // nil when tone matching is off
	lv   *levelTracker

	hops, pos int
}

func newStreamSpectral(cfg SpectralConfig, toneStrength float64, lv *levelTracker) *streamSpectral {
	if cfg.Overshoot < 1 {
		cfg.Overshoot = 1
	}
	if cfg.ReverbTime <= 0 {
		cfg.ReverbTime = 0.45
	}
	s := &streamSpectral{
		fft: dsp.NewRFFT(fftSize), lv: lv,
		in: make([]float64, fftSize), ola: make([]float64, fftSize), frame: make([]float64, fftSize),
		spec: make([]complex128, nBins), pow: make([]float64, nBins), gm: make([]float64, nBins),
		zeros: make([]float64, nBins), ones: make([]float64, nBins),
		nt: newNoiseTracker(),
	}
	for k := range s.ones {
		s.ones[k] = 1
	}
	if cfg.Active() {
		s.gs = newGainState(cfg, SampleRate)
	}
	if toneStrength > 0 {
		s.tone = newLiveTone(toneStrength)
	}
	return s
}

// hop consumes hopSize new samples and produces hopSize finished ones.
func (s *streamSpectral) hop(x, out []float32) {
	copy(s.in, s.in[hopSize:])
	for i, v := range x {
		s.in[fftSize-hopSize+i] = float64(v)
	}
	for i := range s.frame {
		s.frame[i] = s.in[i] * window[i]
	}
	s.fft.Forward(s.frame, s.spec)
	for k, v := range s.spec {
		s.pow[k] = real(v)*real(v) + imag(v)*imag(v)
	}
	if s.hops%2 == 0 { // the noise tracker and tone analysis work on 512-sample strides
		s.nt.add(s.pow)
		if s.tone != nil {
			s.tone.add(s.pow, s.lv.speaking())
		}
	}

	g := s.ones
	if s.gs != nil {
		noise := s.zeros
		if s.nt.valid {
			noise = s.nt.noise
		}
		g = s.gs.update(s.pos, s.pow, noise)
	}
	if s.tone != nil {
		for k := range s.gm {
			s.gm[k] = g[k] * s.tone.gains[k]
		}
		g = s.gm
	}

	for k := range s.spec {
		s.spec[k] *= complex(g[k], 0)
	}
	s.fft.Inverse(s.spec, s.frame)
	for i, v := range s.frame {
		s.ola[i] += v * window[i] * 0.5 // Σ w² == 2 at 75 % overlap
	}
	for i := 0; i < hopSize; i++ {
		out[i] = float32(s.ola[i])
	}
	copy(s.ola, s.ola[hopSize:])
	for i := fftSize - hopSize; i < fftSize; i++ {
		s.ola[i] = 0
	}
	s.hops++
	s.pos += hopSize
}

// control is called once a second to let the slow adaptations catch up.
func (s *streamSpectral) control() {
	if s.tone != nil && s.nt.valid {
		s.tone.update(s.nt.noise)
	}
}

// ---- causal noise tracker -----------------------------------------------------

// noiseTracker is the causal counterpart of estimateNoise: it keeps the last
// two seconds of 512-sample-stride spectra and, every ~85 ms, averages the
// quietest ~10 % of them (judged on speech-band energy). Noise that gets
// quieter is followed at once; noise that gets louder is picked up as the old,
// quieter frames leave the window, i.e. within about two seconds.
type noiseTracker struct {
	spectra [][]float32
	energy  []float32
	scratch []float32
	acc     []float64
	smooth  []float64
	noise   []float64
	pos     int
	filled  int
	since   int
	valid   bool
}

const (
	noiseTrackFrames = noiseHalfWindow // 188 stride frames ≈ 2 s
	noiseMinFrames   = 8               // first estimate after ~85 ms
	noiseEarlyFrames = 60              // below this, take more frames to limit variance
)

func newNoiseTracker() *noiseTracker {
	n := &noiseTracker{
		spectra: make([][]float32, noiseTrackFrames), energy: make([]float32, noiseTrackFrames),
		scratch: make([]float32, noiseTrackFrames),
		acc:     make([]float64, nBins), smooth: make([]float64, nBins), noise: make([]float64, nBins),
	}
	for i := range n.spectra {
		n.spectra[i] = make([]float32, nBins)
	}
	return n
}

func (n *noiseTracker) add(pow []float64) {
	var e float64
	for _, p := range pow[quietLoBin : quietHiBin+1] {
		e += p
	}
	for k, p := range pow {
		n.spectra[n.pos][k] = float32(p)
	}
	n.energy[n.pos] = float32(e)
	n.pos = (n.pos + 1) % noiseTrackFrames
	if n.filled < noiseTrackFrames {
		n.filled++
	}
	n.since++
	if n.since >= gridEvery && n.filled >= noiseMinFrames {
		n.since = 0
		n.estimate()
	}
}

func (n *noiseTracker) estimate() {
	q := quietFraction
	if n.filled < noiseEarlyFrames {
		q = 0.30
	}
	e := n.scratch[:n.filled]
	copy(e, n.energy[:n.filled])
	slices.Sort(e)
	thr := e[min(int(q*float64(n.filled)), n.filled-1)]

	for k := range n.acc {
		n.acc[k] = 0
	}
	count := 0
	for i := 0; i < n.filled; i++ {
		if n.energy[i] > thr {
			continue
		}
		for k, p := range n.spectra[i] {
			n.acc[k] += float64(p)
		}
		count++
	}
	inv := 1 / float64(count)
	for k := range n.smooth { // ±2-bin moving average, scaled to a mean
		lo, hi := max(k-2, 0), min(k+2, nBins-1)
		var s float64
		for _, v := range n.acc[lo : hi+1] {
			s += v
		}
		n.smooth[k] = s / float64(hi-lo+1) * inv
	}
	if !n.valid {
		copy(n.noise, n.smooth)
		n.valid = true
		return
	}
	for k := range n.noise { // damp jitter between successive estimates
		n.noise[k] = 0.7*n.noise[k] + 0.3*n.smooth[k]
	}
}

// ---- live tone matching -----------------------------------------------------------

type binWeight struct {
	k int
	w float64
}

// liveTone learns the speech spectrum while someone talks and slowly steers a
// per-bin gain curve towards the studio target (see tonematch.go). The band
// levels are measured on the 1024-point STFT with fractional bin weights, so
// the narrow low bands are still represented.
type liveTone struct {
	strength float64
	weights  [][]binWeight
	acc      []float64 // exponentially forgotten speech band power
	n        float64   // matching frame count
	forget   float64
	gains    []float64 // linear per-bin gain, starts flat
	updates  int

	speechDB, noiseDB []float64
}

const (
	toneForgetSeconds = 20.0
	toneMinSpeech     = 60.0 // frames (≈ 0.64 s) of speech before the first correction
)

func newLiveTone(strength float64) *liveTone {
	t := &liveTone{
		strength: strength,
		acc:      make([]float64, len(toneCentres)),
		forget:   math.Exp(-1 / (toneForgetSeconds * SampleRate / strideSize)),
		gains:    make([]float64, nBins),
		speechDB: make([]float64, len(toneCentres)),
		noiseDB:  make([]float64, len(toneCentres)),
	}
	for k := range t.gains {
		t.gains[k] = 1
	}
	bin := float64(SampleRate) / fftSize
	for _, fc := range toneCentres {
		lo, hi := fc/math.Pow(2, 1.0/6), fc*math.Pow(2, 1.0/6)
		var ws []binWeight
		for k := 0; k < nBins; k++ {
			a, b := (float64(k)-0.5)*bin, (float64(k)+0.5)*bin
			if overlap := math.Min(b, hi) - math.Max(a, lo); overlap > 0 {
				ws = append(ws, binWeight{k, overlap / bin})
			}
		}
		t.weights = append(t.weights, ws)
	}
	return t
}

func (t *liveTone) bandPower(pow []float64, b int) float64 {
	var e float64
	for _, w := range t.weights[b] {
		e += w.w * pow[w.k]
	}
	return e
}

func (t *liveTone) add(pow []float64, speaking bool) {
	if !speaking {
		return
	}
	for b := range t.acc {
		t.acc[b] = t.acc[b]*t.forget + t.bandPower(pow, b)
	}
	t.n = t.n*t.forget + 1
}

// update recomputes the target correction from what has been heard and moves
// the applied gains a fraction of the way there.
func (t *liveTone) update(noise []float64) bool {
	if t.n < toneMinSpeech {
		return false
	}
	const norm = fftSize * fftSize
	for b := range t.acc {
		t.speechDB[b] = 10 * math.Log10(t.acc[b]/t.n/norm+1e-20)
		t.noiseDB[b] = 10 * math.Log10(t.bandPower(noise, b)/norm+1e-20)
	}
	corr := toneCorrection(t.speechDB, t.speechDB, t.noiseDB, t.strength)
	target := toneGains(corr, SampleRate)
	rate := 0.1
	switch {
	case t.updates < 3:
		rate = 0.5 // find the right ballpark quickly…
	case t.updates < 8:
		rate = 0.25
	} // …then settle
	t.updates++
	for k := range t.gains {
		t.gains[k] = math.Exp((1-rate)*math.Log(t.gains[k]) + rate*math.Log(target[k]))
	}
	return true
}

// streamAnalyzer listens to the signal without touching it: it feeds the noise
// tracker and the tone learner, so tone matching can work in front of a
// neural denoiser that has no STFT of its own.
type streamAnalyzer struct {
	fft   *dsp.RFFT
	in    []float64
	frame []float64
	spec  []complex128
	pow   []float64
	pend  [hopSize]float32
	n     int
	hops  int
	nt    *noiseTracker
	tone  *liveTone
	lv    *levelTracker
}

func newStreamAnalyzer(toneStrength float64, lv *levelTracker) *streamAnalyzer {
	return &streamAnalyzer{
		fft: dsp.NewRFFT(fftSize), in: make([]float64, fftSize), frame: make([]float64, fftSize),
		spec: make([]complex128, nBins), pow: make([]float64, nBins),
		nt: newNoiseTracker(), tone: newLiveTone(toneStrength), lv: lv,
	}
}

func (a *streamAnalyzer) push(x float32) {
	a.pend[a.n] = x
	if a.n++; a.n < hopSize {
		return
	}
	a.n = 0
	copy(a.in, a.in[hopSize:])
	for i, v := range a.pend {
		a.in[fftSize-hopSize+i] = float64(v)
	}
	for i := range a.frame {
		a.frame[i] = a.in[i] * window[i]
	}
	a.fft.Forward(a.frame, a.spec)
	if a.hops++; a.hops%2 == 0 {
		for k, v := range a.spec {
			a.pow[k] = real(v)*real(v) + imag(v)*imag(v)
		}
		a.nt.add(a.pow)
		a.tone.add(a.pow, a.lv.speaking())
	}
}

// control re-learns the curve; it reports whether the gains changed.
func (a *streamAnalyzer) control() bool {
	if !a.nt.valid {
		return false
	}
	return a.tone.update(a.nt.noise)
}
