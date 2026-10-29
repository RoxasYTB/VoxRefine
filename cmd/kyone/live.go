package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"kyone/internal/audio"
	"kyone/internal/device"
	"kyone/internal/enhance"
)

func cmdDevices(args []string) error {
	if len(args) > 0 {
		return errors.New("usage: kyone devices")
	}
	mics, err := device.List(device.Capture)
	if err != nil {
		return err
	}
	outs, err := device.List(device.Playback)
	if err != nil {
		return err
	}
	print := func(title string, devs []device.Info, markCable bool) {
		fmt.Println(title)
		if len(devs) == 0 {
			fmt.Println("  (none)")
		}
		for _, d := range devs {
			tags := ""
			if d.Default {
				tags += " [default]"
			}
			if markCable && device.IsVirtualCable(d.Name) {
				tags += " [virtual cable]"
			}
			fmt.Printf("  %s%s\n", d.Name, tags)
		}
	}
	print("Microphones (use with --input):", mics, false)
	fmt.Println()
	print("Outputs (use with --output):", outs, true)
	if device.VirtualCable(outs) == "" {
		fmt.Printf("\nNo virtual cable found: %s.\n", device.InstallHint())
	}
	return nil
}

func cmdLive(args []string) error {
	fs := flag.NewFlagSet("live", flag.ContinueOnError)
	ef := registerEnhanceFlags(fs)
	input := fs.String("input", "", "microphone: part of its device name (default: the system default)")
	output := fs.String("output", "", "output device: part of its name (default: an installed virtual cable)")
	monitor := fs.Bool("monitor", false, "play on the default output to hear the result (wear headphones: no feedback protection)")
	level := fs.Float64("level-db", enhance.LiveSpeechLevelDB, "speech level the automatic level aims for, in dBFS")
	block := fs.Int("block-ms", 10, "audio block length in ms (smaller = lower latency, higher risk of dropouts)")
	file := fs.String("file", "", "simulate: run this audio file through the live engine, in real-time-sized blocks, and write <file>.live.wav (no devices needed)")
	simOut := fs.String("file-out", "", "output of --file (default: <file>.live.wav)")
	quiet := fs.Bool("quiet", false, "no status line")
	fs.BoolVar(quiet, "q", false, "shorthand for --quiet")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: kyone live [options]\n\nProcesses the microphone in real time and sends it to a virtual cable that\nyour chat application can use as its microphone. Ctrl+C stops it.\n\nOptions:")
		fs.PrintDefaults()
	}
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 0 {
		fs.Usage()
		return errors.New("live takes no file arguments")
	}
	opt, err := ef.options(fs)
	if err != nil {
		return err
	}

	if opt.Neural, err = ef.neural(true); err != nil {
		return err
	}
	if *file != "" {
		return simulateLive(*file, *simOut, opt, *level, *block, *quiet)
	}

	out := *output
	if out == "" && !*monitor {
		outs, err := device.List(device.Playback)
		if err != nil {
			return err
		}
		if out = device.VirtualCable(outs); out == "" {
			return fmt.Errorf("no virtual cable found: %s\n(or use --output to name a device, or --monitor to listen on the default output)", device.InstallHint())
		}
	}

	live, err := enhance.NewLive(opt, *level)
	if err != nil {
		return err
	}
	defer live.Close()
	live.SetBlockSize(device.SampleRate * *block / 1000) // fixed-size audio callbacks
	sess, err := device.Start(device.Config{Input: *input, Output: out, BlockMS: *block}, live.Process)
	if err != nil {
		return err
	}
	defer sess.Stop()

	algoMS := 1000 * float64(live.Latency()) / enhance.SampleRate
	fmt.Fprintf(os.Stderr, "kyone live: %s\n", sess.Description())
	engine := "native engine"
	if opt.Neural != nil {
		engine = opt.Neural.Name
	}
	fmt.Fprintf(os.Stderr, "profile %s, %s, algorithmic latency %.0f ms (+ about %d ms of audio buffers)\n", *ef.profile, engine, algoMS, 6**block)
	if out != "" {
		fmt.Fprintln(os.Stderr, "choose that output device as the microphone in your chat application; Ctrl+C to stop")
	} else {
		fmt.Fprintln(os.Stderr, "monitoring on the default output; Ctrl+C to stop")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			fmt.Fprintln(os.Stderr, "\nstopped")
			return nil
		case <-tick.C:
			if *quiet {
				continue
			}
			st := sess.Stats()
			fmt.Fprintf(os.Stderr, "\rmic %s dB → out %s dB   dropouts: %d   ", peakDB(st.InputPeak), peakDB(st.OutputPeak), st.Underruns+st.Overruns)
		}
	}
}

func peakDB(p float32) string {
	if p <= 0 {
		return " -inf"
	}
	return fmt.Sprintf("%5.1f", 20*math.Log10(float64(p)))
}

// simulateLive feeds a file through the live engine exactly as the microphone
// would be (480-sample blocks), compensating the engine's delay so the result
// lines up with the input. It lets the real-time algorithm be judged — and
// measured — without any audio hardware.
func simulateLive(path, outPath string, opt enhance.Options, levelDB float64, blockMS int, quiet bool) error {
	if outPath == "" {
		outPath = strings.TrimSuffix(path, filepath.Ext(path)) + ".live.wav"
	}
	if _, err := os.Stat(outPath); err == nil {
		return fmt.Errorf("%s already exists", outPath)
	}
	in, err := audio.Decode(path, "")
	if err != nil {
		return err
	}
	mono := audio.Resample(in.ToChannels(1), enhance.SampleRate).Ch[0]

	live, err := enhance.NewLive(opt, levelDB)
	if err != nil {
		return err
	}
	defer live.Close()
	block := enhance.SampleRate * blockMS / 1000
	live.SetBlockSize(block)
	d := live.Latency()
	padded := append(append([]float32(nil), mono...), make([]float32, d)...) // flush the delay line
	out := make([]float32, len(padded))
	start := time.Now()
	for s := 0; s < len(padded); s += block {
		e := min(s+block, len(padded))
		live.Process(padded[s:e], out[s:e])
	}
	elapsed := time.Since(start)

	res := &audio.Buffer{Rate: enhance.SampleRate, Ch: [][]float32{out[d:]}}
	if err := writeAtomic(outPath, res, 16); err != nil {
		return err
	}
	if !quiet {
		secs := float64(len(mono)) / enhance.SampleRate
		fmt.Fprintf(os.Stderr, "live simulation of %s: %.1f s processed in %.2f s (%.1f%% of one core), latency %.0f ms\nwrote %s\n",
			path, secs, elapsed.Seconds(), 100*elapsed.Seconds()/secs, 1000*float64(d)/enhance.SampleRate, outPath)
	}
	return nil
}
