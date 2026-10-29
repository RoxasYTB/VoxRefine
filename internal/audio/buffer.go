// Package audio handles decoding, encoding and resampling of voice recordings.
package audio

import "fmt"

// Buffer is de-interleaved floating-point audio, nominally in [-1, 1].
type Buffer struct {
	Rate int
	Ch   [][]float32
}

// NewBuffer allocates a silent buffer.
func NewBuffer(rate, channels, frames int) *Buffer {
	b := &Buffer{Rate: rate, Ch: make([][]float32, channels)}
	for i := range b.Ch {
		b.Ch[i] = make([]float32, frames)
	}
	return b
}

// Channels returns the channel count.
func (b *Buffer) Channels() int { return len(b.Ch) }

// Frames returns the number of samples per channel.
func (b *Buffer) Frames() int {
	if len(b.Ch) == 0 {
		return 0
	}
	return len(b.Ch[0])
}

// Duration returns the length in seconds.
func (b *Buffer) Duration() float64 {
	if b.Rate == 0 {
		return 0
	}
	return float64(b.Frames()) / float64(b.Rate)
}

// Clone returns a deep copy.
func (b *Buffer) Clone() *Buffer {
	c := &Buffer{Rate: b.Rate, Ch: make([][]float32, len(b.Ch))}
	for i, ch := range b.Ch {
		c.Ch[i] = append([]float32(nil), ch...)
	}
	return c
}

// Validate checks that every channel has the same length.
func (b *Buffer) Validate() error {
	if b.Rate <= 0 {
		return fmt.Errorf("invalid sample rate %d", b.Rate)
	}
	if len(b.Ch) == 0 {
		return fmt.Errorf("no audio channels")
	}
	for i, ch := range b.Ch {
		if len(ch) != len(b.Ch[0]) {
			return fmt.Errorf("channel %d has %d frames, channel 0 has %d", i, len(ch), len(b.Ch[0]))
		}
	}
	return nil
}

// ToChannels returns a buffer with exactly n channels (1 or 2).
// Extra channels are folded down evenly into the first two (odd-numbered
// channels left, even-numbered right); a mono source is duplicated for
// stereo; stereo is averaged for mono. The receiver is never modified.
func (b *Buffer) ToChannels(n int) *Buffer {
	src := len(b.Ch)
	if src == n {
		return b
	}
	out := NewBuffer(b.Rate, n, b.Frames())
	switch {
	case n == 1:
		for _, ch := range b.Ch {
			for i, v := range ch {
				out.Ch[0][i] += v
			}
		}
		inv := 1 / float32(src)
		for i := range out.Ch[0] {
			out.Ch[0][i] *= inv
		}
	case n == 2 && src == 1:
		copy(out.Ch[0], b.Ch[0])
		copy(out.Ch[1], b.Ch[0])
	case n == 2:
		var cnt [2]float32
		for c, ch := range b.Ch {
			d := c % 2
			cnt[d]++
			for i, v := range ch {
				out.Ch[d][i] += v
			}
		}
		for d := 0; d < 2; d++ {
			if cnt[d] == 0 {
				continue
			}
			for i := range out.Ch[d] {
				out.Ch[d][i] /= cnt[d]
			}
		}
	default:
		panic("audio: ToChannels supports 1 or 2 channels")
	}
	return out
}
