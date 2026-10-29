package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"

	"kyone/internal/audio"
	"kyone/internal/enhance"
	"kyone/internal/loudness"
)

func cmdAnalyze(args []string) error {
	fs := flag.NewFlagSet("analyze", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "print machine-readable JSON")
	ffmpeg := fs.String("ffmpeg", "", "path to ffmpeg for formats other than WAV/MP3")
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("usage: kyone analyze [--json] <input>")
	}
	in, err := audio.Decode(pos[0], *ffmpeg)
	if err != nil {
		return err
	}
	r := loudness.Measure(in)

	if *asJSON {
		num := func(v float64) *float64 { // JSON has no -Infinity
			if math.IsInf(v, 0) || math.IsNaN(v) {
				return nil
			}
			return &v
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]any{
			"file":            pos[0],
			"duration_s":      in.Duration(),
			"sample_rate":     in.Rate,
			"channels":        in.Channels(),
			"loudness_lufs":   num(r.IntegratedLUFS),
			"loudness_range":  num(r.LRA),
			"true_peak_dbtp":  num(r.TruePeakDB),
			"sample_peak_db":  num(r.SamplePeakDB),
			"rms_db":          num(r.RMSDB),
			"noise_floor_db":  num(r.NoiseFloorDB),
			"speech_level_db": num(r.SpeechLevelDB),
			"snr_db":          num(r.SNR()),
			"clipped_samples": r.Clipped,
		})
	}

	fmt.Printf("%s\n", pos[0])
	fmt.Printf("  duration       %.1f s, %d Hz, %d channel(s)\n", in.Duration(), in.Rate, in.Channels())
	fmt.Printf("  loudness       %s LUFS (range %s LU)\n", db(r.IntegratedLUFS), db(r.LRA))
	fmt.Printf("  peaks          %s dBTP true, %s dBFS sample", db(r.TruePeakDB), db(r.SamplePeakDB))
	if r.Clipped > 0 {
		fmt.Printf("  — %d clipped sample(s)", r.Clipped)
	}
	fmt.Println()
	fmt.Printf("  noise floor    %s dBFS\n", db(r.NoiseFloorDB))
	fmt.Printf("  speech level   %s dBFS\n", db(r.SpeechLevelDB))
	fmt.Printf("  estimated SNR  %s dB  (%s)\n", db(r.SNR()), verdict(r))
	fmt.Printf("\n  `kyone clean` would target %g LUFS at 48 kHz.\n", enhance.DefaultOptions().TargetLUFS)
	return nil
}

func db(v float64) string {
	if math.IsInf(v, 0) || math.IsNaN(v) {
		return "—"
	}
	return fmt.Sprintf("%.1f", v)
}

// verdict turns the SNR estimate into a short human assessment.
func verdict(r loudness.Result) string {
	switch snr := r.SNR(); {
	case math.IsNaN(snr) || math.IsInf(snr, 0):
		return "not enough signal to tell"
	case snr >= 45:
		return "very clean"
	case snr >= 32:
		return "clean"
	case snr >= 22:
		return "some background noise"
	default:
		return "noisy"
	}
}

func cmdProfiles() error {
	for _, p := range enhance.Profiles() {
		marker := " "
		if p.Name == enhance.DefaultProfile {
			marker = "*"
		}
		inSpeech := ""
		if p.SpeechFloorDB > 0 {
			inSpeech = fmt.Sprintf(" (−%.0f dB in speech)", p.SpeechFloorDB)
		}
		fmt.Printf("%s %-9s noise −%.0f dB%s, dereverb %.0f, tone match %.0f, clarity %.0f\n    %s\n",
			marker, p.Name, p.AttenuationLimitDB, inSpeech, p.Dereverb, p.ToneMatch, p.Clarity, p.Description)
	}
	fmt.Println("\n* default. Use --attenuation-limit-db for a custom noise limit (0–100 dB);")
	fmt.Println("  --tone-match, --clarity, --dereverb and --speech-floor-db override a profile's values.")
	return nil
}
