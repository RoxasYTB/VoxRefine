// Package deepfilter connects kyone to DeepFilterNet, a neural noise
// suppressor (https://github.com/Rikorose/DeepFilterNet, MIT or Apache-2.0).
// kyone does not bundle it: files are processed by calling the official
// `deep-filter` command-line tool, and the live mode loads the C library
// (libdf) built from the same project. Both are found at run time.
package deepfilter

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"kyone/internal/audio"
)

// SampleRate is the only rate DeepFilterNet works at.
const SampleRate = 48000

// ExeName is the name of the command-line tool on this platform.
func ExeName() string {
	if runtime.GOOS == "windows" {
		return "deep-filter.exe"
	}
	return "deep-filter"
}

// LibName is the file name of the C library on this platform.
func LibName() string {
	switch runtime.GOOS {
	case "windows":
		return "df.dll"
	case "darwin":
		return "libdf.dylib"
	default:
		return "libdf.so"
	}
}

// DefaultModel is the low-latency model kyone's live mode expects to find next
// to the library (it is in the DeepFilterNet repository under models/).
const DefaultModel = "DeepFilterNet3_ll_onnx.tar.gz"

// searchDirs lists where a helper file is looked for: next to kyone, in a
// "deepfilter" folder next to kyone, and in the working directory.
func searchDirs() []string {
	var dirs []string
	if exe, err := os.Executable(); err == nil {
		d := filepath.Dir(exe)
		dirs = append(dirs, d, filepath.Join(d, "deepfilter"))
	}
	if wd, err := os.Getwd(); err == nil {
		dirs = append(dirs, wd)
	}
	return dirs
}

func firstExisting(name string) string {
	for _, d := range searchDirs() {
		p := filepath.Join(d, name)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
	}
	return ""
}

// FindCLI locates the deep-filter executable: the explicit path, then the
// KYONE_DEEP_FILTER environment variable, then PATH, then next to kyone.
func FindCLI(explicit string) (string, error) {
	for _, p := range []string{explicit, os.Getenv("KYONE_DEEP_FILTER")} {
		if p != "" {
			if _, err := exec.LookPath(p); err != nil {
				return "", fmt.Errorf("deep-filter not usable at %q: %w", p, err)
			}
			return p, nil
		}
	}
	if p, err := exec.LookPath(ExeName()); err == nil {
		return p, nil
	}
	if p := firstExisting(ExeName()); p != "" {
		return p, nil
	}
	return "", fmt.Errorf("deep-filter not found: download it from https://github.com/Rikorose/DeepFilterNet/releases (%s), put it on your PATH or next to kyone, or pass --deepfilter-bin", releaseAsset())
}

// releaseAsset names the release file to download for this platform.
func releaseAsset() string {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "darwin/arm64":
		return "deep-filter-0.5.6-aarch64-apple-darwin"
	case "darwin/amd64":
		return "deep-filter-0.5.6-x86_64-apple-darwin"
	case "windows/amd64":
		return "deep-filter-0.5.6-x86_64-pc-windows-msvc.exe"
	default:
		return "the asset for your platform"
	}
}

// CLI runs the deep-filter command-line tool on whole recordings.
type CLI struct {
	Bin        string // path to deep-filter
	Model      string // optional model tar.gz (default: the tool's built-in model)
	PostFilter bool   // enable the tool's post-filter (stronger, slightly more artefacts)
}

// Denoise removes noise from a mono 48 kHz recording and returns a signal of
// the same length and alignment. attenDB is the maximum noise reduction in dB
// (0 returns the input unchanged, 100 means unlimited).
func (c *CLI) Denoise(x []float32, attenDB float64) ([]float32, error) {
	if attenDB <= 0 {
		return append([]float32(nil), x...), nil
	}
	dir, err := os.MkdirTemp("", "kyone-df-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	in := filepath.Join(dir, "in.wav")
	f, err := os.Create(in)
	if err != nil {
		return nil, err
	}
	err = audio.WriteWAVFloat32(f, &audio.Buffer{Rate: SampleRate, Ch: [][]float32{x}})
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, err
	}

	outDir := filepath.Join(dir, "out")
	args := []string{"-D", "-a", fmt.Sprintf("%g", min(attenDB, 100)), "-o", outDir}
	if c.Model != "" {
		args = append(args, "-m", c.Model)
	}
	if c.PostFilter {
		args = append(args, "--pf")
	}
	args = append(args, in)
	cmd := exec.Command(c.Bin, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if out, err := cmd.Output(); err != nil {
		return nil, fmt.Errorf("deep-filter failed: %v: %s%s", err, bytes.TrimSpace(out), bytes.TrimSpace(stderr.Bytes()))
	}

	rf, err := os.Open(filepath.Join(outDir, "in.wav"))
	if err != nil {
		return nil, fmt.Errorf("deep-filter produced no output: %w", err)
	}
	defer rf.Close()
	res, err := audio.ReadWAV(rf)
	if err != nil {
		return nil, fmt.Errorf("reading deep-filter output: %w", err)
	}
	if res.Rate != SampleRate || res.Channels() != 1 {
		return nil, fmt.Errorf("deep-filter returned %d Hz, %d channel(s), expected 48 kHz mono", res.Rate, res.Channels())
	}
	// -D compensates the delay; keep the length exactly, whatever the tool does
	// with the last partial frame.
	y := res.Ch[0]
	switch {
	case len(y) > len(x):
		y = y[:len(x)]
	case len(y) < len(x):
		y = append(y, make([]float32, len(x)-len(y))...)
	}
	return y, nil
}
