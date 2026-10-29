package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"kyone/internal/audio"
	"kyone/internal/enhance"
)

func cmdClean(args []string) error {
	fs := flag.NewFlagSet("clean", flag.ContinueOnError)
	ef := registerEnhanceFlags(fs)
	target := fs.Float64("target-lufs", -16, "loudness target in LUFS (−16 suits podcasts)")
	mono := fs.Bool("mono", false, "mix the result down to one channel")
	bits := fs.Int("bits", 16, "output bit depth: 16 or 24")
	ffmpeg := fs.String("ffmpeg", "", "path to ffmpeg for formats other than WAV/MP3 (default: from PATH)")
	force := fs.Bool("force", false, "overwrite the output file if it exists")
	quiet := fs.Bool("quiet", false, "only print errors")
	fs.BoolVar(quiet, "q", false, "shorthand for --quiet")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: kyone clean [options] <input> [output]\n\nOptions:")
		fs.PrintDefaults()
	}

	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return err
	}
	if len(pos) < 1 || len(pos) > 2 {
		fs.Usage()
		return errors.New("expected an input file and an optional output file")
	}
	opt, err := ef.options(fs)
	if err != nil {
		return err
	}
	if opt.Neural, err = ef.neural(false); err != nil {
		return err
	}
	opt.TargetLUFS = *target
	opt.Mono = *mono
	if err := opt.Validate(); err != nil {
		return err
	}

	input := pos[0]
	output := defaultOutput(input)
	if len(pos) == 2 {
		output = pos[1]
	}
	if same, _ := sameFile(input, output); same {
		return errors.New("the output file would overwrite the input")
	}
	if _, err := os.Stat(output); err == nil && !*force {
		return fmt.Errorf("%s already exists (use --force to overwrite it)", output)
	}

	logf := func(format string, a ...any) {
		if !*quiet {
			fmt.Fprintf(os.Stderr, format, a...)
		}
	}

	start := time.Now()
	logf("reading %s\n", input)
	in, err := audio.Decode(input, *ffmpeg)
	if err != nil {
		return err
	}
	logf("  %.1f s, %d Hz, %d channel(s)\n", in.Duration(), in.Rate, in.Channels())

	out, rep, err := enhance.Run(in, opt, func(stage string) { logf("  · %s\n", stage) })
	if err != nil {
		return err
	}

	if err := writeAtomic(output, out, *bits); err != nil {
		return err
	}
	if t := rep.ToneSummary(); t != "" {
		logf("\ntone match: %s", t)
	}
	logf("\n%s\n\nwrote %s (%.1f s, %d Hz, %d-bit) in %.1f s\n  [%s]\n",
		rep.Summary(), output, out.Duration(), out.Rate, *bits, time.Since(start).Seconds(), rep.Timings())
	return nil
}

// writeAtomic writes the WAV to a temporary file next to path and renames it,
// so a failure never leaves a truncated output behind.
func writeAtomic(path string, b *audio.Buffer, bits int) (err error) {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".kyone-*.wav")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
		}
	}()
	if err = audio.WriteWAV(tmp, b, bits); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	if err = os.Chmod(tmp.Name(), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func sameFile(a, b string) (bool, error) {
	ia, err := os.Stat(a)
	if err != nil {
		return false, err
	}
	ib, err := os.Stat(b)
	if err != nil {
		return false, err
	}
	return os.SameFile(ia, ib), nil
}
