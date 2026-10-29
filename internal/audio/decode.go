package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/hajimehoshi/go-mp3"
)

// Decode reads an audio file into a Buffer. WAV and MP3 are decoded natively;
// anything else (M4A, AAC, FLAC, OGG, video containers…) goes through ffmpeg,
// which is looked up as ffmpegPath ("" means "ffmpeg" on PATH).
// Only the first audio track is used.
func Decode(path, ffmpegPath string) (*Buffer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic [12]byte
	n, _ := io.ReadFull(f, magic[:])
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	var buf *Buffer
	switch kind := sniff(magic[:n]); kind {
	case "wav":
		buf, err = ReadWAV(f)
		if err != nil {
			// Exotic WAV encodings (ADPCM, A-law…) can still go through ffmpeg.
			if fb, ferr := decodeFFmpeg(path, ffmpegPath); ferr == nil {
				return normalise(fb)
			}
		}
	case "mp3":
		buf, err = decodeMP3(f)
	default:
		buf, err = decodeFFmpeg(path, ffmpegPath)
	}
	if err != nil {
		return nil, err
	}
	return normalise(buf)
}

// normalise folds >2 channels down to stereo and collapses stereo whose two
// channels are bit-identical (MP3 decoders always emit stereo) into mono.
func normalise(b *Buffer) (*Buffer, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	if b.Frames() == 0 {
		return nil, errors.New("audio file contains no samples")
	}
	if b.Channels() > 2 {
		b = b.ToChannels(2)
	}
	if b.Channels() == 2 {
		same := true
		for i, v := range b.Ch[0] {
			if v != b.Ch[1][i] {
				same = false
				break
			}
		}
		if same {
			b = &Buffer{Rate: b.Rate, Ch: b.Ch[:1]}
		}
	}
	return b, nil
}

func sniff(head []byte) string {
	switch {
	case len(head) >= 12 && string(head[0:4]) == "RIFF" && string(head[8:12]) == "WAVE":
		return "wav"
	case len(head) >= 3 && string(head[0:3]) == "ID3":
		return "mp3"
	case len(head) >= 2 && head[0] == 0xFF && head[1]&0xE0 == 0xE0 && head[1]&0x06 == 0x02:
		// MPEG frame sync with layer III bits (ADTS AAC has them cleared).
		return "mp3"
	}
	return ""
}

func decodeMP3(r io.Reader) (*Buffer, error) {
	dec, err := mp3.NewDecoder(r)
	if err != nil {
		return nil, fmt.Errorf("mp3: %w", err)
	}
	raw, err := io.ReadAll(dec) // 16-bit little-endian stereo
	if err != nil {
		return nil, fmt.Errorf("mp3: %w", err)
	}
	frames := len(raw) / 4
	b := NewBuffer(dec.SampleRate(), 2, frames)
	for i := 0; i < frames; i++ {
		b.Ch[0][i] = float32(int16(binary.LittleEndian.Uint16(raw[i*4:]))) / 32768
		b.Ch[1][i] = float32(int16(binary.LittleEndian.Uint16(raw[i*4+2:]))) / 32768
	}
	return b, nil
}

func decodeFFmpeg(path, ffmpegPath string) (*Buffer, error) {
	if ffmpegPath == "" {
		ffmpegPath = "ffmpeg"
	}
	bin, err := exec.LookPath(ffmpegPath)
	if err != nil {
		return nil, fmt.Errorf("%s is not a WAV or MP3 file and ffmpeg was not found (install ffmpeg or pass --ffmpeg)", path)
	}
	cmd := exec.Command(bin, "-v", "error", "-nostdin", "-i", path,
		"-map", "0:a:0", "-vn", "-f", "wav", "-acodec", "pcm_f32le", "pipe:1")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffmpeg failed to decode %s: %v: %s", path, err, bytes.TrimSpace(stderr.Bytes()))
	}
	return ReadWAV(bytes.NewReader(out))
}
