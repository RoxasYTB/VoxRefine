package audio

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
)

const (
	wavFormatPCM       = 1
	wavFormatFloat     = 3
	wavFormatExtension = 0xFFFE
)

// ReadWAV decodes a RIFF/WAVE stream: integer PCM (8, 16, 24, 32 bits) and
// IEEE float (32, 64 bits), including WAVE_FORMAT_EXTENSIBLE.
func ReadWAV(r io.Reader) (*Buffer, error) {
	br := bufio.NewReader(r)
	var hdr [12]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, fmt.Errorf("wav: reading header: %w", err)
	}
	if string(hdr[0:4]) != "RIFF" || string(hdr[8:12]) != "WAVE" {
		return nil, errors.New("wav: not a RIFF/WAVE file")
	}

	var (
		format, channels, bits uint16
		rate                   uint32
		haveFmt                bool
	)
	for {
		var ch [8]byte
		if _, err := io.ReadFull(br, ch[:]); err != nil {
			return nil, errors.New("wav: no data chunk found")
		}
		id := string(ch[0:4])
		size := binary.LittleEndian.Uint32(ch[4:8])
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, errors.New("wav: fmt chunk too short")
			}
			body := make([]byte, size+size%2)
			if _, err := io.ReadFull(br, body); err != nil {
				return nil, fmt.Errorf("wav: reading fmt chunk: %w", err)
			}
			format = binary.LittleEndian.Uint16(body[0:2])
			channels = binary.LittleEndian.Uint16(body[2:4])
			rate = binary.LittleEndian.Uint32(body[4:8])
			bits = binary.LittleEndian.Uint16(body[14:16])
			if format == wavFormatExtension && size >= 26 {
				format = binary.LittleEndian.Uint16(body[24:26])
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, errors.New("wav: data chunk before fmt chunk")
			}
			return decodeWAVData(br, size, format, int(channels), int(rate), int(bits))
		default:
			if _, err := io.CopyN(io.Discard, br, int64(size)+int64(size%2)); err != nil {
				return nil, errors.New("wav: truncated file")
			}
		}
	}
}

func decodeWAVData(r io.Reader, size uint32, format uint16, channels, rate, bits int) (*Buffer, error) {
	if channels < 1 || rate < 1 {
		return nil, fmt.Errorf("wav: invalid format (%d channels, %d Hz)", channels, rate)
	}
	switch {
	case format == wavFormatPCM && (bits == 8 || bits == 16 || bits == 24 || bits == 32):
	case format == wavFormatFloat && (bits == 32 || bits == 64):
	default:
		return nil, fmt.Errorf("wav: unsupported encoding (format %d, %d bits)", format, bits)
	}
	frameSize := bits / 8 * channels

	// Streaming WAVs (ffmpeg to a pipe) declare 0 or 0xFFFFFFFF: read to EOF.
	known := size != 0 && size != 0xFFFFFFFF
	buf := &Buffer{Rate: rate, Ch: make([][]float32, channels)}
	remaining := int64(size)
	if known {
		for c := range buf.Ch {
			buf.Ch[c] = make([]float32, 0, int(size)/frameSize)
		}
	}

	// Decode block by block so the raw bytes are never held in memory at once.
	raw := make([]byte, frameSize*(1<<15))
	for {
		want := len(raw)
		if known {
			want = int(min(int64(want), remaining/int64(frameSize)*int64(frameSize)))
			if want == 0 {
				break
			}
		}
		n, err := io.ReadFull(r, raw[:want])
		frames := n / frameSize
		for c := range buf.Ch {
			start := len(buf.Ch[c])
			buf.Ch[c] = append(buf.Ch[c], make([]float32, frames)...)
			decodeSamples(buf.Ch[c][start:], raw, c, channels, format, bits)
		}
		remaining -= int64(n)
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			break // end of stream, or a truncated file: keep what we have
		}
		if err != nil {
			return nil, fmt.Errorf("wav: reading samples: %w", err)
		}
	}
	return buf, nil
}

// decodeSamples converts channel c of the interleaved block raw into dst.
func decodeSamples(dst []float32, raw []byte, c, channels int, format uint16, bits int) {
	bytesPerSample := bits / 8
	frameSize := bytesPerSample * channels
	for i := range dst {
		p := raw[i*frameSize+c*bytesPerSample:]
		switch {
		case format == wavFormatFloat && bits == 32:
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(p))
		case format == wavFormatFloat:
			dst[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(p)))
		case bits == 8:
			dst[i] = (float32(p[0]) - 128) / 128
		case bits == 16:
			dst[i] = float32(int16(binary.LittleEndian.Uint16(p))) / 32768
		case bits == 24:
			dst[i] = float32(int32(p[0])|int32(p[1])<<8|int32(int8(p[2]))<<16) / 8388608
		default:
			dst[i] = float32(int32(binary.LittleEndian.Uint32(p))) / 2147483648
		}
	}
}

// WriteWAV encodes b as little-endian integer PCM with the given bit depth
// (16 or 24). Samples are dithered with TPDF noise and clipped to full scale.
// The dither is seeded, so identical input produces identical bytes.
func WriteWAV(w io.Writer, b *Buffer, bits int) error {
	if err := b.Validate(); err != nil {
		return err
	}
	if bits != 16 && bits != 24 {
		return fmt.Errorf("wav: unsupported output bit depth %d (use 16 or 24)", bits)
	}
	channels, frames := b.Channels(), b.Frames()
	bytesPerSample := bits / 8
	dataSize := uint64(frames) * uint64(channels) * uint64(bytesPerSample)
	if dataSize > math.MaxUint32-36 {
		return errors.New("wav: output exceeds the 4 GiB WAV limit")
	}

	bw := bufio.NewWriterSize(w, 1<<16)
	hdr := make([]byte, 44)
	copy(hdr[0:], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:], uint32(36+dataSize))
	copy(hdr[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(hdr[16:], 16)
	binary.LittleEndian.PutUint16(hdr[20:], wavFormatPCM)
	binary.LittleEndian.PutUint16(hdr[22:], uint16(channels))
	binary.LittleEndian.PutUint32(hdr[24:], uint32(b.Rate))
	binary.LittleEndian.PutUint32(hdr[28:], uint32(b.Rate*channels*bytesPerSample))
	binary.LittleEndian.PutUint16(hdr[32:], uint16(channels*bytesPerSample))
	binary.LittleEndian.PutUint16(hdr[34:], uint16(bits))
	copy(hdr[36:], "data")
	binary.LittleEndian.PutUint32(hdr[40:], uint32(dataSize))
	if _, err := bw.Write(hdr); err != nil {
		return err
	}

	rng := rand.New(rand.NewSource(1))
	scale := float64(int64(1) << (bits - 1))
	maxV, minV := scale-1, -scale
	frame := make([]byte, channels*bytesPerSample)
	for i := 0; i < frames; i++ {
		for c := 0; c < channels; c++ {
			v := float64(b.Ch[c][i]) * scale
			v += rng.Float64() - rng.Float64() // TPDF dither, ±1 LSB
			q := math.Round(v)
			if q > maxV {
				q = maxV
			} else if q < minV {
				q = minV
			}
			s := int32(q)
			o := c * bytesPerSample
			frame[o] = byte(s)
			frame[o+1] = byte(s >> 8)
			if bits == 24 {
				frame[o+2] = byte(s >> 16)
			}
		}
		if _, err := bw.Write(frame); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// WriteWAVFloat32 encodes b as 32-bit IEEE-float WAV, without quantisation or
// dither. It is meant for hand-offs between processing tools.
func WriteWAVFloat32(w io.Writer, b *Buffer) error {
	if err := b.Validate(); err != nil {
		return err
	}
	channels, frames := b.Channels(), b.Frames()
	dataSize := uint64(frames) * uint64(channels) * 4
	if dataSize > math.MaxUint32-36 {
		return errors.New("wav: output exceeds the 4 GiB WAV limit")
	}
	bw := bufio.NewWriterSize(w, 1<<16)
	hdr := make([]byte, 44)
	copy(hdr[0:], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:], uint32(36+dataSize))
	copy(hdr[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(hdr[16:], 16)
	binary.LittleEndian.PutUint16(hdr[20:], wavFormatFloat)
	binary.LittleEndian.PutUint16(hdr[22:], uint16(channels))
	binary.LittleEndian.PutUint32(hdr[24:], uint32(b.Rate))
	binary.LittleEndian.PutUint32(hdr[28:], uint32(b.Rate*channels*4))
	binary.LittleEndian.PutUint16(hdr[32:], uint16(channels*4))
	binary.LittleEndian.PutUint16(hdr[34:], 32)
	copy(hdr[36:], "data")
	binary.LittleEndian.PutUint32(hdr[40:], uint32(dataSize))
	if _, err := bw.Write(hdr); err != nil {
		return err
	}
	var sample [4]byte
	for i := 0; i < frames; i++ {
		for c := 0; c < channels; c++ {
			binary.LittleEndian.PutUint32(sample[:], math.Float32bits(b.Ch[c][i]))
			if _, err := bw.Write(sample[:]); err != nil {
				return err
			}
		}
	}
	return bw.Flush()
}
