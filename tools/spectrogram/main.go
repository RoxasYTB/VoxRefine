// Command spectrogram renders one or more WAV files as stacked spectrograms
// (0–12 kHz, time left to right) in a single PNG, for comparing a recording
// before and after processing. It is a development helper, not part of kyone.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

const (
	fftN     = 1024
	hop      = 480 // 100 columns per second at 48 kHz
	maxFreq  = 12000.0
	panelGap = 6
)

func main() {
	floor := flag.Float64("floor", -110, "dBFS mapped to black")
	ceil := flag.Float64("ceil", -30, "dBFS mapped to white")
	flag.Parse()
	if flag.NArg() < 2 {
		fmt.Fprintln(os.Stderr, "usage: spectrogram [options] out.png in1.wav [in2.wav ...]")
		os.Exit(2)
	}
	if err := run(flag.Arg(0), flag.Args()[1:], *floor, *ceil); err != nil {
		fmt.Fprintln(os.Stderr, "spectrogram:", err)
		os.Exit(1)
	}
}

func run(outPath string, inputs []string, floor, ceil float64) error {
	var panels []*image.RGBA
	width := 0
	for _, p := range inputs {
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		b, err := audio.ReadWAV(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		img := render(audio.Resample(b.ToChannels(1), 48000).Ch[0], floor, ceil)
		panels = append(panels, img)
		width = max(width, img.Bounds().Dx())
	}
	height := 0
	for _, p := range panels {
		height += p.Bounds().Dy() + panelGap
	}
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	for i := range canvas.Pix {
		canvas.Pix[i] = 255
	}
	y := 0
	for _, p := range panels {
		for py := 0; py < p.Bounds().Dy(); py++ {
			copy(canvas.Pix[(y+py)*canvas.Stride:], p.Pix[py*p.Stride:py*p.Stride+p.Bounds().Dx()*4])
		}
		y += p.Bounds().Dy() + panelGap
	}
	o, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer o.Close()
	return png.Encode(o, canvas)
}

func render(x []float32, floor, ceil float64) *image.RGBA {
	cols := len(x)/hop + 1
	rows := int(maxFreq / 48000 * fftN)
	img := image.NewRGBA(image.Rect(0, 0, cols, rows))
	fft := dsp.NewRFFT(fftN)
	buf := make([]float64, fftN)
	spec := make([]complex128, fft.Bins())
	for c := 0; c < cols; c++ {
		start := c*hop - fftN/2
		for i := range buf {
			j := start + i
			if j >= 0 && j < len(x) {
				buf[i] = float64(x[j]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/fftN))
			} else {
				buf[i] = 0
			}
		}
		fft.Forward(buf, spec)
		for r := 0; r < rows; r++ {
			// Amplitude of a full-scale sine is 0 dBFS (Hann window gain 0.5).
			mag := math.Hypot(real(spec[r]), imag(spec[r])) / (fftN / 4)
			db := 20 * math.Log10(mag+1e-12)
			img.Set(c, rows-1-r, heat((db-floor)/(ceil-floor)))
		}
	}
	return img
}

// heat maps 0..1 to a black → purple → orange → white gradient.
func heat(t float64) color.RGBA {
	t = math.Min(math.Max(t, 0), 1)
	stops := []struct {
		at      float64
		r, g, b float64
	}{{0, 0, 0, 10}, {0.3, 70, 10, 110}, {0.6, 200, 60, 70}, {0.85, 250, 170, 50}, {1, 255, 255, 230}}
	for i := 1; i < len(stops); i++ {
		if t <= stops[i].at {
			a, b := stops[i-1], stops[i]
			f := (t - a.at) / (b.at - a.at)
			return color.RGBA{uint8(a.r + f*(b.r-a.r)), uint8(a.g + f*(b.g-a.g)), uint8(a.b + f*(b.b-a.b)), 255}
		}
	}
	return color.RGBA{255, 255, 230, 255}
}
