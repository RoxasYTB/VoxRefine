// Command kyone cleans up and enhances voice recordings: it removes noise and
// room reverb, polishes the tone and brings the level to a podcast standard.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var version = "0.1.0"

const usage = `kyone %s — make a voice recording sound clean and professional

Usage:
  kyone clean   [options] <input> [output]   enhance a recording
  kyone analyze [--json] <input>             measure loudness, peaks and noise
  kyone live    [options]                    enhance the microphone in real time
  kyone devices                              list microphones, outputs and virtual cables
  kyone profiles                             list the presets
  kyone version

"kyone <input>" is a shortcut for "kyone clean <input>". The default output is
<input>.enhanced.wav (48 kHz WAV) next to the input; existing files are never
overwritten unless --force is given.

Input: WAV and MP3 natively; M4A, AAC, FLAC, OGG, video… through ffmpeg.

Run "kyone clean -h" for the full list of options.
`

func main() {
	if err := run(os.Args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintln(os.Stderr, "kyone:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, usage, version)
		os.Exit(2)
	}
	switch cmd := args[0]; cmd {
	case "clean":
		return cmdClean(args[1:])
	case "analyze":
		return cmdAnalyze(args[1:])
	case "live":
		return cmdLive(args[1:])
	case "devices":
		return cmdDevices(args[1:])
	case "profiles":
		return cmdProfiles()
	case "version", "-v", "--version":
		fmt.Println("kyone", version)
		return nil
	case "help", "-h", "--help":
		fmt.Printf(usage, version)
		return nil
	default:
		if _, err := os.Stat(cmd); err == nil && !strings.HasPrefix(cmd, "-") {
			return cmdClean(args)
		}
		return fmt.Errorf("unknown command %q (try \"kyone help\")", cmd)
	}
}

// parseInterspersed parses flags that may appear before, between or after the
// positional arguments, which the standard flag package does not allow.
func parseInterspersed(fs *flag.FlagSet, args []string) ([]string, error) {
	var positional []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		rest := fs.Args()
		if len(rest) == 0 {
			return positional, nil
		}
		positional = append(positional, rest[0])
		args = rest[1:]
	}
}

// defaultOutput returns <dir>/<name>.enhanced.wav for an input path.
func defaultOutput(input string) string {
	base := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
	return filepath.Join(filepath.Dir(input), base+".enhanced.wav")
}
