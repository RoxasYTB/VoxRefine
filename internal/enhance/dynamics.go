package enhance

import (
	"math"
	"slices"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// Compressor settings. The threshold is placed relative to the measured
// speech level, so the same settings suit quiet and loud recordings.
const (
	compRatio      = 2.5
	compKneeDB     = 8.0
	compAttackMS   = 8.0
	compReleaseMS  = 140.0
	compDetectMS   = 25.0 // RMS window of the level detector
	compOffsetDB   = -3.0 // threshold relative to the speech level
	compMaxReduceB = 12.0
)

// compressor evens out the level of loud and soft words with a soft-knee,
// stereo-linked downward compressor. Its threshold sits relative to the active
// speech level in dBFS (see loudness.Result.SpeechLevelDB), which can be
// updated while running.
type compressor struct {
	threshold        float64
	detect, att, rel float64
	power, gr, maxGR float64
	enabled          bool
}

func newCompressor(rate, speechDB float64) *compressor {
	c := &compressor{detect: pole(rate, compDetectMS), att: pole(rate, compAttackMS), rel: pole(rate, compReleaseMS)}
	c.setSpeechDB(speechDB)
	return c
}

// setSpeechDB moves the threshold; an unknown (infinite or NaN) level disables
// the compressor until a real one is supplied.
func (c *compressor) setSpeechDB(speechDB float64) {
	c.enabled = !(math.IsInf(speechDB, 0) || math.IsNaN(speechDB))
	c.threshold = speechDB + compOffsetDB
}

// process handles one time step: x holds one sample per channel, in place.
func (c *compressor) process(x []float32) {
	var p float64
	for _, v := range x {
		p += float64(v) * float64(v)
	}
	c.power = c.detect*c.power + (1-c.detect)*p/float64(len(x))
	if !c.enabled {
		return
	}
	levelDB := 10 * math.Log10(c.power+1e-14)

	// Soft-knee static curve: how many dB to take off.
	over := levelDB - c.threshold
	var target float64
	switch {
	case 2*over < -compKneeDB:
		target = 0
	case 2*over > compKneeDB:
		target = (1 - 1/compRatio) * over
	default:
		t := over + compKneeDB/2
		target = (1 - 1/compRatio) * t * t / (2 * compKneeDB)
	}
	target = math.Min(target, compMaxReduceB)
	c.gr = follow(c.gr, target, c.att, c.rel)
	if c.gr > c.maxGR {
		c.maxGR = c.gr
	}
	if c.gr > 0.001 {
		g := float32(math.Pow(10, -c.gr/20))
		for i := range x {
			x[i] *= g
		}
	}
}

// compress runs the compressor over a whole buffer and returns the largest
// gain reduction applied, in dB.
func compress(b *audio.Buffer, speechDB float64) float64 {
	c := newCompressor(float64(b.Rate), speechDB)
	eachFrame(b, c.process)
	return c.maxGR
}

// limiterLookahead is the look-ahead (and ramp) length of the true-peak
// limiter in milliseconds.
const limiterLookahead = 10.0

// limitEvent is a sample whose true peak exceeds the ceiling, with the gain
// that would bring it exactly to the ceiling.
type limitEvent struct {
	i    int
	need float32
}

// limit is an offline look-ahead true-peak limiter. The gain required at each
// over-ceiling sample is widened with a sliding minimum and smoothed with a
// moving average of the same length, which guarantees that the gain at every
// peak is at least as low as required. Over-ceiling samples are sparse, so
// only the neighbourhood of each cluster of them is ever materialised and the
// rest of the signal is not touched. It returns the largest gain reduction
// applied, in dB.
func limit(b *audio.Buffer, ceilingDB float64) float64 {
	n := b.Frames()
	if n == 0 {
		return 0
	}
	L := max(int(math.Round(limiterLookahead/1000*float64(b.Rate))), 2)
	events := findOvers(b, float32(dsp.FromDB(ceilingDB)))

	minG := float32(1)
	for start := 0; start < len(events); {
		// A cluster is a run of events close enough for their ramps to overlap.
		end := start + 1
		for end < len(events) && events[end].i-events[end-1].i <= 2*L+2 {
			end++
		}
		lo := max(events[start].i-L, 0)
		hi := min(events[end-1].i+L+2, n)
		need := make([]float32, hi-lo)
		for i := range need {
			need[i] = 1
		}
		for _, e := range events[start:end] {
			need[e.i-lo] = min(need[e.i-lo], e.need)
		}
		g := limiterGain(need, L)
		for _, ch := range b.Ch {
			for i, gain := range g {
				ch[lo+i] *= gain
			}
		}
		for _, gain := range g {
			minG = min(minG, gain)
		}
		start = end
	}
	return -dsp.DB(float64(minG))
}

// findOvers scans the 4x-oversampled peak envelope, linked across channels,
// block by block and lists the samples above the ceiling.
func findOvers(b *audio.Buffer, ceiling float32) []limitEvent {
	const block = 1 << 15
	n := b.Frames()
	env := make([]float32, block+2*dsp.TruePeakMargin)
	linked := make([]float32, block+2*dsp.TruePeakMargin)
	var events []limitEvent
	for a := 0; a < n; a += block {
		e := min(a+block, n)
		lo, hi := max(a-dsp.TruePeakMargin, 0), min(e+dsp.TruePeakMargin, n)
		for c, ch := range b.Ch {
			dsp.TruePeakEnvelopeInto(env[:hi-lo], ch[lo:hi])
			if c == 0 {
				copy(linked, env[:hi-lo])
				continue
			}
			for i, v := range env[:hi-lo] {
				linked[i] = max(linked[i], v)
			}
		}
		for i := a; i < e; i++ {
			if p := linked[i-lo]; p > ceiling {
				events = append(events, limitEvent{i, ceiling / p})
			}
		}
	}
	return events
}

// limiterGain turns the required-gain curve need (<= 1) into a smooth gain
// curve g with g[j] <= need[j] everywhere. Samples outside the slice are
// taken to need no reduction, except that the first sample's value is
// replicated backwards so that a peak in the first L samples is still covered.
func limiterGain(need []float32, L int) []float32 {
	n := len(need)
	// m[i] = min(need[i : i+L]): the dip is announced L samples early.
	rev := append([]float32(nil), need...)
	slices.Reverse(rev)
	m := trailingMin(rev, L)
	slices.Reverse(m)

	// g[j] = mean(m[j-L+1 : j+1]) turns the stepped dip into a ramp. Every
	// m[i] in that window is <= need[j], so g[j] <= need[j] and the ceiling holds.
	g := make([]float32, n)
	sum := float64(L) * float64(m[0]) // the window before the start replicates m[0]
	for j := 0; j < n; j++ {
		sum += float64(m[j])
		if old := j - L; old >= 0 {
			sum -= float64(m[old])
		} else {
			sum -= float64(m[0])
		}
		g[j] = float32(sum / float64(L))
	}
	return g
}

// trailingMin returns out[i] = min(x[max(0,i-L+1) : i+1]) in O(n).
func trailingMin(x []float32, L int) []float32 {
	out := make([]float32, len(x))
	dq := make([]int, 0, L)
	for i, v := range x {
		for len(dq) > 0 && x[dq[len(dq)-1]] >= v {
			dq = dq[:len(dq)-1]
		}
		dq = append(dq, i)
		if dq[0] <= i-L {
			dq = dq[1:]
		}
		out[i] = x[dq[0]]
	}
	return out
}
