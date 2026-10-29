package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func sine(rate int, freq, amp, seconds float64) *Buffer {
	n := int(float64(rate) * seconds)
	b := NewBuffer(rate, 1, n)
	for i := range b.Ch[0] {
		b.Ch[0][i] = float32(amp * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return b
}

func rms(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(len(x)))
}

func TestWAVRoundTrip(t *testing.T) {
	for _, bits := range []int{16, 24} {
		in := sine(48000, 440, 0.5, 0.25)
		in.Ch = append(in.Ch, append([]float32(nil), in.Ch[0]...))
		for i := range in.Ch[1] {
			in.Ch[1][i] *= 0.5
		}
		var buf bytes.Buffer
		if err := WriteWAV(&buf, in, bits); err != nil {
			t.Fatal(err)
		}
		out, err := ReadWAV(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if out.Rate != 48000 || out.Channels() != 2 || out.Frames() != in.Frames() {
			t.Fatalf("%d-bit: got %d Hz %d ch %d frames", bits, out.Rate, out.Channels(), out.Frames())
		}
		tol := 2.0 / math.Pow(2, float64(bits-1))
		for c := range in.Ch {
			for i := range in.Ch[c] {
				if d := math.Abs(float64(in.Ch[c][i] - out.Ch[c][i])); d > tol {
					t.Fatalf("%d-bit ch%d[%d]: diff %g > %g", bits, c, i, d, tol)
				}
			}
		}
	}
}

func TestWriteWAVClips(t *testing.T) {
	in := NewBuffer(48000, 1, 4)
	copy(in.Ch[0], []float32{2, -2, 1, -1})
	var buf bytes.Buffer
	if err := WriteWAV(&buf, in, 16); err != nil {
		t.Fatal(err)
	}
	out, err := ReadWAV(&buf)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out.Ch[0] {
		if v > 1 || v < -1 {
			t.Fatalf("sample %d out of range: %v", i, v)
		}
	}
}

func TestResamplePreservesToneAndLevel(t *testing.T) {
	for _, tc := range []struct{ from, to int }{{44100, 48000}, {16000, 48000}, {96000, 48000}, {48000, 44100}} {
		in := sine(tc.from, 1000, 0.5, 1)
		out := Resample(in, tc.to)
		if out.Rate != tc.to {
			t.Fatalf("rate %d", out.Rate)
		}
		wantFrames := tc.to
		if d := out.Frames() - wantFrames; d < -1 || d > 1 {
			t.Errorf("%d→%d: %d frames, want ≈%d", tc.from, tc.to, out.Frames(), wantFrames)
		}
		// Compare the middle of the output with an ideal sine at the new rate.
		lo, hi := tc.to/10, tc.to*9/10
		var errSum float64
		for i := lo; i < hi; i++ {
			want := 0.5 * math.Sin(2*math.Pi*1000*float64(i)/float64(tc.to))
			d := float64(out.Ch[0][i]) - want
			errSum += d * d
		}
		rmsErr := math.Sqrt(errSum / float64(hi-lo))
		if rmsErr > 0.002 { // ≈ -54 dB relative to a 0.35 RMS tone
			t.Errorf("%d→%d: RMS error %.5f too high", tc.from, tc.to, rmsErr)
		}
	}
}

func TestResampleRejectsAliasing(t *testing.T) {
	// A 30 kHz tone at 96 kHz must vanish when downsampling to 48 kHz.
	in := sine(96000, 30000, 0.5, 0.5)
	out := Resample(in, 48000)
	mid := out.Ch[0][out.Frames()/4 : out.Frames()*3/4]
	if r := rms(mid); r > 0.005 {
		t.Fatalf("aliased energy RMS %.4f", r)
	}
}

func TestToChannels(t *testing.T) {
	in := NewBuffer(48000, 2, 2)
	in.Ch[0][0], in.Ch[1][0] = 1, 0
	in.Ch[0][1], in.Ch[1][1] = 0.5, 0.5
	mono := in.ToChannels(1)
	if mono.Ch[0][0] != 0.5 || mono.Ch[0][1] != 0.5 {
		t.Fatalf("mono mix %v", mono.Ch[0])
	}
	if in.Channels() != 2 {
		t.Fatal("input modified")
	}
	st := mono.ToChannels(2)
	if st.Channels() != 2 || st.Ch[1][0] != 0.5 {
		t.Fatalf("stereo dup %v", st.Ch)
	}
}

func TestNormaliseCollapsesDualMono(t *testing.T) {
	in := sine(44100, 220, 0.3, 0.1)
	in.Ch = append(in.Ch, append([]float32(nil), in.Ch[0]...))
	out, err := normalise(in)
	if err != nil {
		t.Fatal(err)
	}
	if out.Channels() != 1 {
		t.Fatalf("got %d channels", out.Channels())
	}
}

// floatWAV builds a 32-bit float stereo WAV whose data chunk declares dataSize
// (which may lie, as in streamed or truncated files).
func floatWAV(frames int, dataSize uint32, truncateTo int) []byte {
	var b bytes.Buffer
	le := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("RIFF")
	le(uint32(36 + frames*8))
	b.WriteString("WAVEfmt ")
	le(uint32(16))
	le(uint16(3)) // IEEE float
	le(uint16(2))
	le(uint32(48000))
	le(uint32(48000 * 8))
	le(uint16(8))
	le(uint16(32))
	b.WriteString("data")
	le(dataSize)
	for i := 0; i < frames; i++ {
		le(float32(i) / 100000)
		le(-float32(i) / 100000)
	}
	out := b.Bytes()
	if truncateTo > 0 {
		out = out[:truncateTo]
	}
	return out
}

func TestReadWAVStreamedAndTruncated(t *testing.T) {
	const frames = 70000 // more than one decode block
	for _, tc := range []struct {
		name string
		data []byte
		want int
	}{
		{"declared size", floatWAV(frames, frames*8, 0), frames},
		{"streamed size 0xFFFFFFFF", floatWAV(frames, 0xFFFFFFFF, 0), frames},
		{"streamed size 0", floatWAV(frames, 0, 0), frames},
		{"truncated mid-frame", floatWAV(frames, frames*8, 44+1000*8+3), 1000},
	} {
		b, err := ReadWAV(bytes.NewReader(tc.data))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if b.Frames() != tc.want || b.Channels() != 2 || b.Rate != 48000 {
			t.Fatalf("%s: got %d frames, %d ch, %d Hz; want %d frames", tc.name, b.Frames(), b.Channels(), b.Rate, tc.want)
		}
		if b.Ch[0][999] != 999.0/100000 || b.Ch[1][999] != -999.0/100000 {
			t.Fatalf("%s: wrong samples %v %v", tc.name, b.Ch[0][999], b.Ch[1][999])
		}
	}
}

func TestReadWAVRejectsGarbage(t *testing.T) {
	for _, data := range [][]byte{nil, []byte("not a wav file at all"), floatWAV(10, 80, 20)} {
		if _, err := ReadWAV(bytes.NewReader(data)); err == nil {
			t.Errorf("accepted %d bytes of garbage", len(data))
		}
	}
}
