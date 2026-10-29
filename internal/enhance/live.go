package enhance

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"kyone/internal/dsp"
)

// This file is the real-time (streaming, causal) version of the chain. It
// shares the stages of the offline pipeline — the same gain estimator, the
// same de-esser, clarity and compressor — and replaces what needs the whole
// file (noise tracking, tone analysis, loudness normalisation, the limiter's
// look-ahead) with causal equivalents that learn from the past as audio flows.

// Live is a mono, 48 kHz real-time voice enhancer. Feed it the microphone in
// blocks of any size; every call returns as many processed samples as it was
// given, delayed by Latency(). It is not safe for concurrent use and does not
// allocate once running, so it can be driven from an audio callback.
type Live struct {
	hp      dsp.Cascade
	spec    *streamSpectral
	neural  StreamDenoiser // external neural denoiser, instead of spec
	nFrame  int
	nPend   []float32
	nOut    []float32
	ana     *streamAnalyzer // tone learning in front of the neural denoiser
	fir     *adaptiveFIR    // and its tone curve as a filter
	deess   *deEsser
	eq      dsp.Cascade
	clar    *clarifier
	lev     *leveler
	comp    *compressor
	lim     *streamLimiter
	lvRaw   *levelTracker // after the high-pass: gates the tone analysis
	lvMid   *levelTracker // before levelling: positions clarity, leveler, compressor
	one     [1]float32
	pend    [hopSize]float32
	pending int
	hopOut  [hopSize]float32
	outQ    fifo
	latency int
	prefill int // samples of silence queued at the start (block alignment)
	started bool

	levelDB    float64 // speech level the leveler aims for
	useLeveler bool
	sinceCtl   int
}

// LiveSpeechLevelDB is the default speech level the leveler aims for (the
// 90th percentile of 50 ms RMS, the same measure as the offline "speech level").
const LiveSpeechLevelDB = -20.0

const (
	liveControlEvery = SampleRate // seconds between control updates, in samples
	liveWarmupBlocks = 20         // 1 s of level statistics before the level-dependent stages start
)

// NewLive builds a live enhancer. opt supplies the same settings as the
// offline chain (noise limit, dereverb, tone, clarity, EQ, de-esser,
// compressor, ceiling); opt.Normalize switches the leveler on, aiming for
// levelDB (0 selects LiveSpeechLevelDB). The sample rate is fixed at 48 kHz.
func NewLive(opt Options, levelDB float64) (*Live, error) {
	check := opt
	if check.TargetLUFS == 0 {
		check.TargetLUFS = -16 // the live chain levels to levelDB instead
	}
	if err := check.Validate(); err != nil {
		return nil, err
	}
	if levelDB == 0 {
		levelDB = LiveSpeechLevelDB
	}
	if levelDB > -6 || levelDB < -50 {
		return nil, errors.New("speech level must be between -50 and -6 dBFS")
	}
	rate := float64(SampleRate)
	l := &Live{
		levelDB:    levelDB,
		useLeveler: opt.Normalize,
		lvRaw:      newLevelTracker(rate),
		lvMid:      newLevelTracker(rate),
		deess:      nil,
		lev:        newLeveler(rate),
		comp:       newCompressor(rate, math.Inf(-1)),
		lim:        newStreamLimiter(rate, opt.CeilingDB),
		outQ:       newFifo(1 << 14),
	}
	if opt.HighPassHz > 0 {
		l.hp = dsp.ButterworthHighPass(rate, opt.HighPassHz, 4)
	}
	if opt.DeEss {
		l.deess = newDeEsser(rate, 1)
	}
	l.eq = toneChain(rate, opt)
	if opt.Clarity > 0 {
		l.clar = newClarifier(rate, 1, opt.Clarity/100)
	}
	if !opt.Compress {
		l.comp = nil
	}

	frame := hopSize
	if opt.Neural != nil && opt.AttenuationLimitDB > 0 {
		// Neural denoising in place of the native spectral stage. kyone's own
		// dereverb is not available here (it would need its own STFT and add
		// its delay); tone matching works as a delay-free filter.
		if opt.Neural.Open == nil {
			return nil, fmt.Errorf("%s cannot run in real time", opt.Neural.Name)
		}
		d, err := opt.Neural.Open(opt.AttenuationLimitDB)
		if err != nil {
			return nil, err
		}
		l.neural = d
		l.nFrame = d.FrameLength()
		l.nPend, l.nOut = make([]float32, l.nFrame), make([]float32, l.nFrame)
		if opt.ToneMatch > 0 {
			l.ana = newStreamAnalyzer(opt.ToneMatch/100, l.lvRaw)
			l.fir = newAdaptiveFIR(toneFIRTaps, rate)
		}
		frame = l.nFrame
		l.latency = d.Latency()
	} else {
		cfg := SpectralConfig{
			AttenuationDB: opt.AttenuationLimitDB,
			Overshoot:     opt.Overshoot,
			SpeechFloorDB: opt.SpeechFloorDB,
			Dereverb:      opt.Dereverb / 100,
		}
		if cfg.Active() || opt.ToneMatch > 0 {
			l.spec = newStreamSpectral(cfg, opt.ToneMatch/100, l.lvRaw)
			l.latency = fftSize - hopSize
		}
	}

	// The output FIFO starts with one frame of silence so that a block can
	// always be answered in full, whatever its alignment with the internal
	// frame (SetBlockSize trims it when the block size is known).
	l.prefill = frame
	if l.neural == nil && l.spec == nil {
		l.prefill = 0 // nothing is buffered
	}
	l.outQ.pushZeros(l.prefill)
	l.latency += l.prefill + l.lim.latency()
	return l, nil
}

// Close releases the neural denoiser, if any.
func (l *Live) Close() {
	if l.neural != nil {
		l.neural.Close()
		l.neural = nil
	}
}

// SetBlockSize declares that Process will always be called with blocks of
// exactly n samples (as an audio callback does). The start-up silence that
// guards against misaligned blocks can then shrink, saving up to one frame of
// latency — all of it when n is a multiple of the frame (10 ms blocks with
// the 10 ms neural frames). Call it before the first Process; it does nothing
// afterwards.
func (l *Live) SetBlockSize(n int) {
	frame := hopSize
	if l.neural != nil {
		frame = l.nFrame
	}
	if n <= 0 || l.started || l.prefill == 0 {
		return
	}
	need := frame - gcd(n, frame) // the worst case of pending samples
	if drop := l.prefill - need; drop > 0 {
		l.outQ.dropFront(drop)
		l.prefill -= drop
		l.latency -= drop
	}
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// Latency is the pure delay between input and output in samples (at 48 kHz).
// Filters add a few more samples of group delay (well under a millisecond).
func (l *Live) Latency() int { return l.latency }

// Process enhances in and writes len(in) samples to out, which must be at
// least as long. The output is the input delayed by Latency().
func (l *Live) Process(in, out []float32) {
	l.started = true
	for _, x := range in {
		if l.hp != nil {
			x = float32(l.hp.Process(float64(x)))
		}
		l.lvRaw.add(x)
		switch {
		case l.neural != nil:
			if l.ana != nil {
				l.ana.push(x)
			}
			l.nPend[l.pending] = x
			l.pending++
			if l.pending == l.nFrame {
				l.neural.Process(l.nPend, l.nOut)
				for _, y := range l.nOut {
					if l.fir != nil {
						y = float32(l.fir.process(float64(y)))
					}
					l.post(y)
				}
				l.pending = 0
			}
		case l.spec == nil:
			l.post(x)
		default:
			l.pend[l.pending] = x
			l.pending++
			if l.pending == hopSize {
				l.spec.hop(l.pend[:], l.hopOut[:])
				for _, y := range l.hopOut {
					l.post(y)
				}
				l.pending = 0
			}
		}
		// Control updates fire at exact sample counts, so the output does not
		// depend on how the caller slices the stream into blocks.
		if l.sinceCtl++; l.sinceCtl >= liveControlEvery {
			l.sinceCtl = 0
			l.control()
		}
	}
	l.outQ.pop(out[:len(in)])
}

// post runs one sample through the per-sample stages and queues the result.
func (l *Live) post(y float32) {
	l.one[0] = y
	if l.deess != nil {
		l.deess.process(l.one[:])
	}
	if len(l.eq) > 0 {
		l.one[0] = float32(l.eq.Process(float64(l.one[0])))
	}
	if l.clar != nil {
		l.clar.process(l.one[:])
	}
	l.lvMid.add(l.one[0])
	l.lev.process(l.one[:])
	if l.comp != nil {
		l.comp.process(l.one[:])
	}
	l.outQ.push(l.lim.process(l.one[0]))
}

// control runs once a second: it turns the level statistics into stage
// settings (clarity gate, leveler gain, compressor threshold, tone curve).
func (l *Live) control() {
	noise, speech, ok := l.lvMid.estimates()
	if !ok {
		return
	}
	if l.clar != nil {
		l.clar.setLevels(noise, speech)
	}
	gain := 0.0
	if l.useLeveler {
		l.lev.aim(l.levelDB-speech, speech-noise)
		gain = l.lev.gainDB
	}
	if l.comp != nil {
		l.comp.setSpeechDB(speech + gain)
	}
	if l.spec != nil {
		l.spec.control()
	}
	if l.ana != nil && l.ana.control() {
		l.fir.set(minPhaseFIR(l.ana.tone.gains, toneFIRTaps))
	}
}

// fifo is a growable ring buffer of samples that does not allocate once it is
// big enough for the largest block.
type fifo struct {
	buf        []float32
	head, size int
}

func newFifo(capacity int) fifo { return fifo{buf: make([]float32, capacity)} }

func (f *fifo) grow() {
	nb := make([]float32, len(f.buf)*2)
	for i := 0; i < f.size; i++ {
		nb[i] = f.buf[(f.head+i)%len(f.buf)]
	}
	f.buf, f.head = nb, 0
}

func (f *fifo) push(x float32) {
	if f.size == len(f.buf) {
		f.grow()
	}
	f.buf[(f.head+f.size)%len(f.buf)] = x
	f.size++
}

// dropFront discards the n oldest samples.
func (f *fifo) dropFront(n int) {
	n = min(n, f.size)
	f.head = (f.head + n) % len(f.buf)
	f.size -= n
}

func (f *fifo) pushZeros(n int) {
	for i := 0; i < n; i++ {
		f.push(0)
	}
}

// pop fills dst, padding with silence if the queue ran short (it never should).
func (f *fifo) pop(dst []float32) {
	for i := range dst {
		if f.size == 0 {
			dst[i] = 0
			continue
		}
		dst[i] = f.buf[f.head]
		f.head = (f.head + 1) % len(f.buf)
		f.size--
	}
}

// ---- level statistics -------------------------------------------------------

// levelTracker keeps the last 20 s of 50 ms RMS levels and reports their 10th
// and 90th percentiles: the noise floor and the active speech level, as in the
// offline analysis, but updated as audio arrives.
type levelTracker struct {
	win     int
	acc     float64
	n       int
	ring    []float64 // block levels in dB
	pos     int
	filled  int
	scratch []float64
	last    float64 // level of the latest complete block, dB
	noise   float64
	speech  float64
	ok      bool
	sinceUp int
}

const levelBlocks = 400 // 20 s of 50 ms blocks

func newLevelTracker(rate float64) *levelTracker {
	return &levelTracker{
		win:     int(0.05 * rate),
		ring:    make([]float64, levelBlocks),
		scratch: make([]float64, levelBlocks),
		last:    math.Inf(-1),
	}
}

func (t *levelTracker) add(x float32) {
	t.acc += float64(x) * float64(x)
	t.n++
	if t.n < t.win {
		return
	}
	t.last = dsp.DB(math.Sqrt(t.acc / float64(t.n)))
	t.acc, t.n = 0, 0
	t.ring[t.pos] = t.last
	t.pos = (t.pos + 1) % len(t.ring)
	if t.filled < len(t.ring) {
		t.filled++
	}
	t.sinceUp++
	if t.sinceUp >= 20 && t.filled >= liveWarmupBlocks { // refresh once a second
		t.sinceUp = 0
		s := t.scratch[:t.filled]
		copy(s, t.ring[:t.filled])
		slices.Sort(s)
		t.noise = quantile(s, 0.10)
		t.speech = quantile(s, 0.90)
		t.ok = true
	}
}

// estimates returns the noise floor and speech level in dBFS; ok is false
// during the first seconds.
func (t *levelTracker) estimates() (noise, speech float64, ok bool) {
	return t.noise, t.speech, t.ok
}

// speaking reports whether the latest block is clearly above the noise floor.
func (t *levelTracker) speaking() bool { return t.ok && t.last > t.noise+10 }

func quantile(sorted []float64, p float64) float64 {
	pos := p * float64(len(sorted)-1)
	lo, hi := int(math.Floor(pos)), int(math.Ceil(pos))
	f := pos - float64(lo)
	return sorted[lo]*(1-f) + sorted[hi]*f
}

// ---- leveler ----------------------------------------------------------------

// leveler is a slow open-loop automatic gain: once a second it is told how far
// the speech level is from the target and moves its gain a fraction of the way,
// at most a couple of dB per step, and only while speech is present, so that
// silence is never amplified.
type leveler struct {
	gainDB  float64 // gain aimed for
	g       float64 // linear gain actually applied (smoothed)
	k       float64
	updates int
}

const (
	levelerMaxDB    = 15.0
	levelerStepDB   = 2.0
	levelerMinSpan  = 15.0 // dB between speech and noise needed to trust the estimate
	levelerFastWarm = 5    // updates that adapt at the faster rate
)

func newLeveler(rate float64) *leveler { return &leveler{g: 1, k: pole(rate, 150)} }

// aim moves the gain towards needed (dB) if the recording currently holds
// speech well above its noise (spanDB = speech − noise).
func (l *leveler) aim(neededDB, spanDB float64) {
	if spanDB < levelerMinSpan {
		return
	}
	rate := 0.25
	if l.updates < levelerFastWarm {
		rate = 0.6
	}
	l.updates++
	// neededDB is relative to the signal *before* the gain, so aim for it directly.
	step := (neededDB - l.gainDB) * rate
	step = math.Min(math.Max(step, -levelerStepDB), levelerStepDB)
	l.gainDB = math.Min(math.Max(l.gainDB+step, -levelerMaxDB), levelerMaxDB)
}

func (l *leveler) process(x []float32) {
	l.g = l.k*l.g + (1-l.k)*dsp.FromDB(l.gainDB)
	for i := range x {
		x[i] *= float32(l.g)
	}
}

// ---- limiter ----------------------------------------------------------------

// streamLimiter is a look-ahead peak limiter that works sample by sample. It
// is the causal form of the offline limiter: the gain needed at each sample is
// widened with a sliding minimum over the next L samples and smoothed with a
// moving average of the same length, so the gain at every peak is at least as
// low as required; the price is a delay of L-1 samples.
type streamLimiter struct {
	L       int
	ceiling float32

	x    []float32 // last L inputs
	xPos int

	dqIdx   []int // monotonic deque of (index, need) for the sliding minimum
	dqVal   []float32
	dqHead  int
	dqCount int

	m    []float32 // last L sliding minima
	mPos int
	sum  float64
	n    int
}

const streamLimiterMS = 3.0

func newStreamLimiter(rate, ceilingDB float64) *streamLimiter {
	L := max(int(streamLimiterMS/1000*rate), 2)
	s := &streamLimiter{
		L: L, ceiling: float32(dsp.FromDB(ceilingDB)),
		x: make([]float32, L), m: make([]float32, L),
		dqIdx: make([]int, L+1), dqVal: make([]float32, L+1),
	}
	for i := range s.m {
		s.m[i] = 1
	}
	s.sum = float64(L)
	return s
}

func (s *streamLimiter) latency() int { return s.L - 1 }

func (s *streamLimiter) process(v float32) float32 {
	s.x[s.xPos] = v
	s.xPos = (s.xPos + 1) % s.L

	need := float32(1)
	if a := float32(math.Abs(float64(v))); a > s.ceiling {
		need = s.ceiling / a
	}
	// Sliding minimum of need over the last L samples.
	size := len(s.dqIdx)
	for s.dqCount > 0 && s.dqVal[(s.dqHead+s.dqCount-1)%size] >= need {
		s.dqCount--
	}
	s.dqIdx[(s.dqHead+s.dqCount)%size] = s.n
	s.dqVal[(s.dqHead+s.dqCount)%size] = need
	s.dqCount++
	if s.dqIdx[s.dqHead] <= s.n-s.L {
		s.dqHead = (s.dqHead + 1) % size
		s.dqCount--
	}
	minNeed := s.dqVal[s.dqHead]
	s.n++

	// Moving average of the minima.
	s.sum += float64(minNeed) - float64(s.m[s.mPos])
	s.m[s.mPos] = minNeed
	s.mPos = (s.mPos + 1) % s.L
	g := float32(s.sum / float64(s.L))

	// The output is the input L-1 samples ago: the oldest value in the ring.
	return s.x[s.xPos] * g
}
