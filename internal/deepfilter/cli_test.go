package deepfilter

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"kyone/internal/audio"
)

// fakeTool writes a shell script that stands in for deep-filter: it records its
// arguments, then writes the input back (or `shorter` samples of it) to the
// output directory under the same name.
func fakeTool(t *testing.T, dir string, mode string) (bin, argsFile string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script stand-in")
	}
	bin = filepath.Join(dir, "deep-filter")
	argsFile = filepath.Join(dir, "args.txt")
	script := `#!/bin/sh
echo "$@" > "` + argsFile + `"
out=""; prev=""; last=""
for a in "$@"; do
  [ "$prev" = "-o" ] && out="$a"
  prev="$a"; last="$a"
done
case "` + mode + `" in
  fail) echo "model exploded" >&2; exit 3 ;;
  nofile) exit 0 ;;
esac
mkdir -p "$out"
cp "$last" "$out/$(basename "$last")"
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin, argsFile
}

func TestCLIDenoisePassesTheRightArgumentsAndKeepsLength(t *testing.T) {
	dir := t.TempDir()
	bin, argsFile := fakeTool(t, dir, "copy")
	x := make([]float32, 48000)
	for i := range x {
		x[i] = float32(i%100) / 1000
	}
	cli := &CLI{Bin: bin, Model: "/models/m.tar.gz", PostFilter: true}
	y, err := cli.Denoise(x, 35.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != len(x) || y[777] != x[777] {
		t.Fatalf("round trip changed the signal (%d samples)", len(y))
	}
	raw, _ := os.ReadFile(argsFile)
	args := string(raw)
	for _, want := range []string{"-D", "-a 35.5", "-o ", "-m /models/m.tar.gz", "--pf", "in.wav"} {
		if !strings.Contains(args, want) {
			t.Errorf("arguments %q lack %q", strings.TrimSpace(args), want)
		}
	}

	// A limit of 0 dB means no noise reduction: the tool is not even called.
	os.Remove(argsFile)
	z, err := cli.Denoise(x, 0)
	if err != nil || len(z) != len(x) {
		t.Fatal(err)
	}
	if _, err := os.Stat(argsFile); err == nil {
		t.Error("the tool ran for a 0 dB limit")
	}
	// Above 100 dB is clamped to the tool's maximum.
	if _, err := cli.Denoise(x, 250); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(argsFile)
	if !strings.Contains(string(raw), "-a 100") {
		t.Errorf("limit not clamped: %s", raw)
	}
}

func TestCLIDenoiseReportsFailures(t *testing.T) {
	dir := t.TempDir()
	x := make([]float32, 4800)

	bin, _ := fakeTool(t, dir, "fail")
	if _, err := (&CLI{Bin: bin}).Denoise(x, 40); err == nil || !strings.Contains(err.Error(), "model exploded") {
		t.Errorf("tool failure not surfaced with its message: %v", err)
	}
	bin2, _ := fakeTool(t, t.TempDir(), "nofile")
	if _, err := (&CLI{Bin: bin2}).Denoise(x, 40); err == nil || !strings.Contains(err.Error(), "no output") {
		t.Errorf("missing output not reported: %v", err)
	}
	if _, err := (&CLI{Bin: filepath.Join(dir, "does-not-exist")}).Denoise(x, 40); err == nil {
		t.Error("a missing binary was accepted")
	}
}

func TestCLIDenoisePadsAndTrimsLength(t *testing.T) {
	// A tool that returns a different number of samples must still give back
	// exactly the input length.
	if runtime.GOOS == "windows" {
		t.Skip("shell-script stand-in")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "deep-filter")
	// Writes a 1000-sample WAV (via the tool's own copy of a prepared file).
	short := filepath.Join(dir, "short.wav")
	f, _ := os.Create(short)
	if err := audio.WriteWAVFloat32(f, &audio.Buffer{Rate: SampleRate, Ch: [][]float32{make([]float32, 1000)}}); err != nil {
		t.Fatal(err)
	}
	f.Close()
	script := "#!/bin/sh\nout=\"\"; prev=\"\"\nfor a in \"$@\"; do [ \"$prev\" = \"-o\" ] && out=\"$a\"; prev=\"$a\"; done\nmkdir -p \"$out\"; cp \"" + short + "\" \"$out/in.wav\"\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	y, err := (&CLI{Bin: bin}).Denoise(make([]float32, 4800), 40)
	if err != nil || len(y) != 4800 {
		t.Fatalf("got %d samples, %v; want 4800", len(y), err)
	}
}

func TestLocatingHelpers(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, LibName())
	model := filepath.Join(dir, DefaultModel)
	for _, p := range []string{lib, model} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Explicit paths win, then the environment.
	if got, err := FindLib(lib); err != nil || got != lib {
		t.Errorf("FindLib explicit: %q, %v", got, err)
	}
	t.Setenv("KYONE_DF_LIB", lib)
	t.Setenv("KYONE_DF_MODEL", model)
	if got, err := FindLib(""); err != nil || got != lib {
		t.Errorf("FindLib env: %q, %v", got, err)
	}
	if got, err := FindModel(""); err != nil || got != model {
		t.Errorf("FindModel env: %q, %v", got, err)
	}
	// Bad paths give errors that say what is wrong.
	if _, err := FindLib(filepath.Join(dir, "missing")); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("FindLib missing: %v", err)
	}
	t.Setenv("KYONE_DF_LIB", "")
	t.Setenv("KYONE_DF_MODEL", "")
	if _, err := FindLib(""); err == nil || !strings.Contains(err.Error(), "cargo build") {
		t.Errorf("the missing-library error should say how to build it: %v", err)
	}
	if _, err := FindModel(""); err == nil || !strings.Contains(err.Error(), DefaultModel) {
		t.Errorf("the missing-model error should name the file: %v", err)
	}
	t.Setenv("PATH", t.TempDir())
	t.Setenv("KYONE_DEEP_FILTER", "")
	if _, err := FindCLI(""); err == nil || !strings.Contains(err.Error(), "github.com/Rikorose/DeepFilterNet/releases") {
		t.Errorf("the missing-tool error should point to the download: %v", err)
	}
	if _, err := OpenLib(lib); err == nil {
		t.Error("a file that is not a library was loaded")
	}
}
