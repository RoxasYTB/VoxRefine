package main

import (
	"flag"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"kyone/internal/audio"
)

// writeNoisyVoice writes a few seconds of a voice-like tone burst train over
// white noise to a WAV file.
func writeNoisyVoice(t *testing.T, path string, rate int) {
	t.Helper()
	rng := rand.New(rand.NewSource(1))
	b := audio.NewBuffer(rate, 1, rate*4)
	for i := range b.Ch[0] {
		tt := float64(i) / float64(rate)
		v := 0.01 * rng.NormFloat64()
		if math.Mod(tt, 1) < 0.6 {
			v += 0.2 * math.Sin(2*math.Pi*180*tt) * math.Sin(math.Pi*math.Mod(tt, 1)/0.6)
		}
		b.Ch[0][i] = float32(v)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := audio.WriteWAV(f, b, 16); err != nil {
		t.Fatal(err)
	}
}

func TestDefaultOutput(t *testing.T) {
	for in, want := range map[string]string{
		"voice.mp3":          "voice.enhanced.wav",
		"/tmp/a b/talk.m4a":  "/tmp/a b/talk.enhanced.wav",
		"dir/noext":          "dir/noext.enhanced.wav",
		"archive.v2.final.w": "archive.v2.final.enhanced.wav",
	} {
		if got := defaultOutput(in); got != want {
			t.Errorf("defaultOutput(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCleanEndToEndAndOverwriteProtection(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "voice.wav")
	writeNoisyVoice(t, in, 44100)

	// Default output name, flags after the positional argument.
	if err := run([]string{"clean", in, "-q", "--profile", "strong"}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "voice.enhanced.wav")
	f, err := os.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	b, err := audio.ReadWAV(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if b.Rate != 48000 || b.Channels() != 1 {
		t.Fatalf("output is %d Hz, %d channel(s)", b.Rate, b.Channels())
	}

	// A second run must refuse to overwrite, --force must allow it.
	err = run([]string{"clean", in, "-q"})
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("expected an overwrite refusal, got %v", err)
	}
	if err := run([]string{"clean", in, out, "-q", "--force", "--bits", "24", "--mono"}); err != nil {
		t.Fatal(err)
	}

	// The shortcut form, and the analyze command, work on the same file.
	if err := run([]string{in, "-q", "--force"}); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"analyze", out}); err != nil {
		t.Fatal(err)
	}

	// No temporary files are left behind.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".kyone-") {
			t.Errorf("leftover temporary file %s", e.Name())
		}
	}
}

func TestCleanRejectsBadArguments(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "voice.wav")
	writeNoisyVoice(t, in, 48000)

	for name, args := range map[string][]string{
		"output equals input":       {"clean", in, in, "-q", "--force"},
		"profile and limit":         {"clean", in, "--profile", "natural", "--attenuation-limit-db", "10"},
		"unknown profile":           {"clean", in, "--profile", "extreme"},
		"limit out of range":        {"clean", in, "--attenuation-limit-db", "120"},
		"bass out of range":         {"clean", in, "--bass-db", "30"},
		"missing input":             {"clean"},
		"nonexistent input":         {"clean", filepath.Join(dir, "missing.wav")},
		"unknown command":           {"frobnicate"},
		"unsupported without tools": {"clean", writeJunk(t, dir, "voice.ogg"), "--ffmpeg", filepath.Join(dir, "no-ffmpeg")},
	} {
		if err := run(args); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func writeJunk(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("OggS\x00\x02 definitely not audio"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParseInterspersed(t *testing.T) {
	fs := newTestFlagSet()
	pos, err := parseInterspersed(fs, []string{"a.wav", "-x", "7", "b.wav", "--y"})
	if err != nil {
		t.Fatal(err)
	}
	if len(pos) != 2 || pos[0] != "a.wav" || pos[1] != "b.wav" {
		t.Fatalf("positional = %v", pos)
	}
}

func newTestFlagSet() *flag.FlagSet {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	fs.Int("x", 0, "")
	fs.Bool("y", false, "")
	return fs
}

func TestLiveSimulationEndToEnd(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "voice.wav")
	writeNoisyVoice(t, in, 44100)

	if err := run([]string{"live", "--file", in, "--profile", "balanced", "-q"}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "voice.live.wav")
	f, err := os.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	b, err := audio.ReadWAV(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if b.Rate != 48000 || b.Channels() != 1 {
		t.Fatalf("simulation output is %d Hz, %d channel(s)", b.Rate, b.Channels())
	}
	if want := 4 * 48000; b.Frames() < want-2 || b.Frames() > want+2 {
		t.Errorf("simulation output has %d frames, want ≈%d (the delay must be compensated)", b.Frames(), want)
	}
	// It refuses to overwrite.
	if err := run([]string{"live", "--file", in, "-q"}); err == nil {
		t.Error("simulation overwrote an existing file")
	}
	// Live takes no positional arguments, and rejects a bad level.
	if err := run([]string{"live", "stray.wav"}); err == nil {
		t.Error("stray argument accepted")
	}
	if err := run([]string{"live", "--file", in, "--file-out", filepath.Join(dir, "x.wav"), "--level-db", "-2"}); err == nil {
		t.Error("speech level -2 dBFS accepted")
	}
}

func TestEngineFlag(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "voice.wav")
	writeNoisyVoice(t, in, 48000)

	if err := run([]string{"clean", in, "--engine", "bogus", "-q"}); err == nil || !strings.Contains(err.Error(), "unknown engine") {
		t.Errorf("unknown engine: %v", err)
	}
	t.Setenv("PATH", t.TempDir())
	t.Setenv("KYONE_DEEP_FILTER", "")
	err := run([]string{"clean", in, "--engine", "deepfilter", "-q"})
	if err == nil || !strings.Contains(err.Error(), "deep-filter not found") {
		t.Errorf("missing tool should be explained: %v", err)
	}
	err = run([]string{"live", "--file", in, "--engine", "deepfilter", "--deepfilter-lib", filepath.Join(dir, "nope.dylib"), "-q"})
	if err == nil || !strings.Contains(err.Error(), "library not found") {
		t.Errorf("missing library should be explained: %v", err)
	}
}
