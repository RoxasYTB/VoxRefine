// Command compare scores a processed recording against its clean reference:
// scale-invariant SDR overall and inside speech, and the residual level in
// the reference's pauses. It is a development helper, not part of kyone.
//
// The two files must be sample-aligned (kyone introduces no delay). Use the
// chain options that keep the level, e.g. --no-normalize --no-compress.
package main

import (
	"fmt"
	"math"
	"os"

	"kyone/internal/audio"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: compare reference.wav test.wav [test2.wav ...]")
		os.Exit(2)
	}
	ref, err := load(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "compare:", err)
		os.Exit(1)
	}
	active := activity(ref, 0.05, -55)
	fmt.Printf("%-28s %10s %12s %14s\n", "file", "SI-SDR dB", "speech SDR", "pause level")
	for _, p := range os.Args[2:] {
		x, err := load(p)
		if err != nil {
			fmt.Fprintln(os.Stderr, "compare:", err)
			os.Exit(1)
		}
		n := min(len(x), len(ref))
		all := make([]bool, n)
		for i := range all {
			all[i] = true
		}
		fmt.Printf("%-28s %10.1f %12.1f %11.1f dB\n", short(p),
			sisdr(x[:n], ref[:n], all, true), sisdr(x[:n], ref[:n], active[:n], true), rmsDB(x[:n], active[:n], false))
	}
	fmt.Printf("%-28s %10s %12s %11.1f dB\n", "(reference)", "", "", rmsDB(ref, active, false))
}

func load(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := audio.ReadWAV(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return audio.Resample(b.ToChannels(1), 48000).Ch[0], nil
}

func short(p string) string {
	if len(p) > 28 {
		return "…" + p[len(p)-27:]
	}
	return p
}

// activity marks samples inside frames whose RMS in the reference exceeds thresholdDB.
func activity(ref []float32, sec, thresholdDB float64) []bool {
	win := int(sec * 48000)
	out := make([]bool, len(ref))
	for s := 0; s+win <= len(ref); s += win {
		var e float64
		for _, v := range ref[s : s+win] {
			e += float64(v) * float64(v)
		}
		on := 10*math.Log10(e/float64(win)+1e-20) > thresholdDB
		for i := s; i < s+win; i++ {
			out[i] = on
		}
	}
	return out
}

// sisdr is the scale-invariant signal-to-distortion ratio over the samples where sel == want.
func sisdr(x, ref []float32, sel []bool, want bool) float64 {
	var xr, rr float64
	for i := range x {
		if sel[i] == want {
			xr += float64(x[i]) * float64(ref[i])
			rr += float64(ref[i]) * float64(ref[i])
		}
	}
	a := xr / rr
	var tgt, errE float64
	for i := range x {
		if sel[i] == want {
			t := a * float64(ref[i])
			e := float64(x[i]) - t
			tgt += t * t
			errE += e * e
		}
	}
	return 10 * math.Log10(tgt/errE)
}

func rmsDB(x []float32, sel []bool, want bool) float64 {
	var s float64
	var n int
	for i, v := range x {
		if sel[i] == want {
			s += float64(v) * float64(v)
			n++
		}
	}
	if n == 0 {
		return math.Inf(-1)
	}
	return 10 * math.Log10(s/float64(n)+1e-20)
}
