package enhance

import (
	"math"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// De-esser parameters. Sibilance is judged by how much of the short-term
// power sits above deEssCutoff; when that share exceeds deEssThreshold the
// band is turned down at deEssRatio, but never by more than deEssMaxDB.
const (
	deEssCutoff    = 5500.0
	deEssThreshold = -10.0 // dB, share of power above the cutoff
	deEssRatio     = 3.0
	deEssMaxDB     = 7.0
	deEssFloorDB   = -55.0 // ignore passages quieter than this (noise)
)

// deEsser attenuates sibilants by splitting the signal with a Linkwitz-Riley
// (4th-order) crossover at deEssCutoff and turning only the high band down:
// out = low + g·high. The two bands are in phase, so low + high is flat and
// scaling the high band attenuates it cleanly (a plain "x − k·highpass(x)"
// would boost around the cutoff because of the filter's phase shift). The
// detector is linked across channels so the stereo image does not wobble.
type deEsser struct {
	lows, highs                    []dsp.Cascade
	lo, hi                         []float64
	pTot, pHi, gr                  float64 // smoothed powers and gain reduction in dB
	powAtt, powRel, gAtt, gRel, fl float64
}

func newDeEsser(rate float64, nCh int) *deEsser {
	d := &deEsser{
		lows: make([]dsp.Cascade, nCh), highs: make([]dsp.Cascade, nCh),
		lo: make([]float64, nCh), hi: make([]float64, nCh),
		powAtt: pole(rate, 1), powRel: pole(rate, 12),
		gAtt: pole(rate, 1.5), gRel: pole(rate, 25),
		fl: math.Pow(10, deEssFloorDB/10),
	}
	for c := 0; c < nCh; c++ {
		d.lows[c] = dsp.Cascade{dsp.LowPass(rate, deEssCutoff, math.Sqrt2/2), dsp.LowPass(rate, deEssCutoff, math.Sqrt2/2)}
		d.highs[c] = dsp.Cascade{dsp.HighPass(rate, deEssCutoff, math.Sqrt2/2), dsp.HighPass(rate, deEssCutoff, math.Sqrt2/2)}
	}
	return d
}

// process handles one time step: x holds one sample per channel, in place.
func (d *deEsser) process(x []float32) {
	nCh := len(x)
	var tot, hiPow float64
	for c := 0; c < nCh; c++ {
		v := float64(x[c])
		d.lo[c] = d.lows[c].Process(v)
		d.hi[c] = d.highs[c].Process(v)
		tot += v * v
		hiPow += d.hi[c] * d.hi[c]
	}
	d.pTot = follow(d.pTot, tot/float64(nCh), d.powAtt, d.powRel)
	d.pHi = follow(d.pHi, hiPow/float64(nCh), d.powAtt, d.powRel)

	target := 0.0
	if d.pTot > d.fl {
		share := 10 * math.Log10(d.pHi/d.pTot+1e-12)
		if over := share - deEssThreshold; over > 0 {
			target = math.Min((1-1/deEssRatio)*over, deEssMaxDB)
		}
	}
	d.gr = follow(d.gr, target, d.gAtt, d.gRel)
	g := math.Pow(10, -d.gr/20)
	for c := 0; c < nCh; c++ {
		x[c] = float32(d.lo[c] + g*d.hi[c])
	}
}

// deEss runs the de-esser over a whole buffer.
func deEss(b *audio.Buffer) {
	d := newDeEsser(float64(b.Rate), b.Channels())
	eachFrame(b, d.process)
}

// follow is a one-pole envelope follower with separate attack and release
// coefficients (closer to 1 means slower).
func follow(prev, x, attack, release float64) float64 {
	k := release
	if x > prev {
		k = attack
	}
	return k*prev + (1-k)*x
}
