package enhance

import (
	"math"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// Clarity settings. Both lifts only act while a voice is speaking, so they
// never raise the noise floor between words.
const (
	clarityPresenceHz = 3200.0
	clarityPresenceQ  = 0.9
	clarityPresenceDB = 5.0 // presence lift at full strength
	clarityAirHz      = 6500.0
	clarityAirDB      = 5.0 // top-end lift at full strength
)

// clarifier makes a voice more present and open. While speech is active it
//
//   - lifts a bell around 3.2 kHz (articulation), and
//   - lifts everything above 6.5 kHz (air), through a Linkwitz-Riley split so
//     the two bands stay in phase; this is only applied to voiced passages
//     (low-mid energy over high energy), so fricatives are not brightened and
//     the de-esser's work is not undone.
//
// Activity is the level above the noise floor (noiseDB and speechDB as
// measured by loudness.Levels), so the lifts fade out into pauses and noise.
// The levels can be updated while running.
type clarifier struct {
	presence []*dsp.Biquad
	lows     []dsp.Cascade
	highs    []dsp.Cascade
	lf, hf   dsp.Cascade

	kPres, kAir                  float64
	pAll, pLF, pHF               float64
	att, rel, levelAtt, levelRel float64
	enabled                      bool
	onFloor, span                float64
}

func newClarifier(rate float64, nCh int, strength float64) *clarifier {
	c := &clarifier{
		presence: make([]*dsp.Biquad, nCh), lows: make([]dsp.Cascade, nCh), highs: make([]dsp.Cascade, nCh),
		att: pole(rate, 3), rel: pole(rate, 100),
		// The activity detector must fall back to the noise floor quickly after
		// a phrase (a slow release would keep lifting the noise in the pause).
		levelAtt: pole(rate, 3), levelRel: pole(rate, 20),
		kPres: dsp.FromDB(clarityPresenceDB*strength) - 1,
		kAir:  dsp.FromDB(clarityAirDB*strength) - 1,
	}
	for i := 0; i < nCh; i++ {
		c.presence[i] = dsp.BandPass(rate, clarityPresenceHz, clarityPresenceQ)
		c.lows[i] = dsp.Cascade{dsp.LowPass(rate, clarityAirHz, math.Sqrt2/2), dsp.LowPass(rate, clarityAirHz, math.Sqrt2/2)}
		c.highs[i] = dsp.Cascade{dsp.HighPass(rate, clarityAirHz, math.Sqrt2/2), dsp.HighPass(rate, clarityAirHz, math.Sqrt2/2)}
	}
	// Sidechain band-passes: voiced energy (300–1500 Hz) against sibilant
	// energy (4–8 kHz).
	c.lf = dsp.Cascade{dsp.HighPass(rate, 300, 0.707), dsp.LowPass(rate, 1500, 0.707)}
	c.hf = dsp.Cascade{dsp.HighPass(rate, 4000, 0.707), dsp.LowPass(rate, 8000, 0.707)}
	return c
}

// setLevels supplies the noise floor and speech level in dBFS. With less than
// 10 dB between them (or unknown levels) the lifts are held off: there is no
// headroom to tell speech from noise.
func (c *clarifier) setLevels(noiseDB, speechDB float64) {
	c.enabled = !(math.IsInf(noiseDB, 0) || math.IsInf(speechDB, 0) || math.IsNaN(noiseDB) || math.IsNaN(speechDB)) && speechDB-noiseDB >= 10
	c.onFloor = noiseDB + 6
	c.span = math.Max(speechDB-6-c.onFloor, 6)
}

// process handles one time step: x holds one sample per channel, in place.
func (c *clarifier) process(x []float32) {
	nCh := len(x)
	var m float64
	for _, v := range x {
		m += float64(v)
	}
	m /= float64(nCh)
	c.pAll = follow(c.pAll, m*m, c.levelAtt, c.levelRel)
	l, h := c.lf.Process(m), c.hf.Process(m)
	c.pLF = follow(c.pLF, l*l, c.att, c.rel)
	c.pHF = follow(c.pHF, h*h, c.att, c.rel)
	if !c.enabled {
		return
	}

	active := clamp01((10*math.Log10(c.pAll+1e-14) - c.onFloor) / c.span)
	voiced := clamp01((10*math.Log10((c.pLF+1e-14)/(c.pHF+1e-14)) + 3) / 9)
	airGain := 1 + c.kAir*active*voiced

	for i := 0; i < nCh; i++ {
		v := float64(x[i])
		// lo + hi is an all-pass version of x, so the split is transparent
		// when the air gain is 1 and scaling hi cleanly lifts the top end.
		lo, hi := c.lows[i].Process(v), c.highs[i].Process(v)
		x[i] = float32(lo + airGain*hi + c.kPres*active*c.presence[i].Process(v))
	}
}

// clarity runs the clarity stage over a whole buffer. strength is 0..1; noiseDB
// and speechDB come from loudness.Levels.
func clarity(b *audio.Buffer, strength, noiseDB, speechDB float64) {
	if strength <= 0 {
		return
	}
	c := newClarifier(float64(b.Rate), b.Channels(), strength)
	c.setLevels(noiseDB, speechDB)
	if !c.enabled {
		return
	}
	eachFrame(b, c.process)
}

func clamp01(x float64) float64 { return math.Min(math.Max(x, 0), 1) }
