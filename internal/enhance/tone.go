package enhance

import (
	"math"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

// Corner frequencies of the tone controls. The shelves match VoxRefine's EQ.
const (
	bassHz     = 150
	presenceHz = 3200
	trebleHz   = 4000
)

// highPass removes rumble and DC with a 4th-order Butterworth filter.
func highPass(b *audio.Buffer, hz float64) {
	if hz <= 0 {
		return
	}
	for _, ch := range b.Ch {
		dsp.ButterworthHighPass(float64(b.Rate), hz, 4).ProcessFloat32(ch)
	}
}

// toneChain builds the biquad cascade of the manual tone controls (bass shelf,
// presence bell, treble shelf, --eq bands); nil when none is set.
func toneChain(rate float64, o Options) dsp.Cascade {
	var chain dsp.Cascade
	if o.BassDB != 0 {
		chain = append(chain, dsp.LowShelf(rate, bassHz, o.BassDB))
	}
	if o.PresenceDB != 0 {
		chain = append(chain, dsp.Peaking(rate, presenceHz, 0.8, o.PresenceDB))
	}
	if o.TrebleDB != 0 {
		chain = append(chain, dsp.HighShelf(rate, trebleHz, o.TrebleDB))
	}
	for _, band := range o.EQ {
		// Keep the corner below Nyquist if a low sample rate was requested.
		hz := math.Min(band.Hz, rate*0.45)
		switch band.Kind {
		case "low":
			chain = append(chain, dsp.LowShelf(rate, hz, band.GainDB))
		case "high":
			chain = append(chain, dsp.HighShelf(rate, hz, band.GainDB))
		default:
			q := band.Q
			if q == 0 {
				q = 0.8
			}
			chain = append(chain, dsp.Peaking(rate, hz, q, band.GainDB))
		}
	}
	return chain
}

// shapeTone applies the manual tone controls. Peaks the boost may create are
// handled by the limiter at the end of the chain.
func shapeTone(b *audio.Buffer, o Options) {
	chain := toneChain(float64(b.Rate), o)
	if len(chain) == 0 {
		return
	}
	for _, ch := range b.Ch {
		chain.Clone().ProcessFloat32(ch)
	}
}
