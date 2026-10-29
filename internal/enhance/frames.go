package enhance

import (
	"math"

	"kyone/internal/audio"
)

// eachFrame calls fn once per time step with that step's samples of every
// channel and stores back whatever fn leaves in the slice. It lets the
// sample-by-sample stages, written as stateful processors for live use, run
// over a whole buffer too.
func eachFrame(b *audio.Buffer, fn func(x []float32)) {
	tmp := make([]float32, b.Channels())
	for i := 0; i < b.Frames(); i++ {
		for c := range tmp {
			tmp[c] = b.Ch[c][i]
		}
		fn(tmp)
		for c := range tmp {
			b.Ch[c][i] = tmp[c]
		}
	}
}

// pole returns the one-pole smoothing coefficient for a time constant in ms.
func pole(rate, ms float64) float64 { return math.Exp(-1 / (ms / 1000 * rate)) }
