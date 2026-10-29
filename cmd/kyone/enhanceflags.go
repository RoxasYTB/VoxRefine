package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"kyone/internal/deepfilter"
	"kyone/internal/enhance"
)

// enhanceFlags are the options that shape the sound, shared by `clean` (files)
// and `live` (microphone).
type enhanceFlags struct {
	profile                          *string
	limit, dereverb, speechFloor     *float64
	highpass, bass, presence, treble *float64
	toneMatch, clarity, ceiling      *float64
	noDeess, noCompress, noNormalize *bool
	eq                               eqFlags

	engine, dfBin, dfLib, dfModel *string
	dfPostFilter                  *bool
}

func registerEnhanceFlags(fs *flag.FlagSet) *enhanceFlags {
	f := &enhanceFlags{}
	f.profile = fs.String("profile", enhance.DefaultProfile, "noise-reduction preset: "+enhance.ProfileNames())
	f.limit = fs.Float64("attenuation-limit-db", -1, "custom maximum noise reduction, 0–100 dB (0 = off); replaces --profile")
	f.dereverb = fs.Float64("dereverb", -1, "room-reverb suppression 0–100 (0 = off); default depends on the profile")
	f.highpass = fs.Float64("highpass", 80, "remove rumble below this frequency in Hz (0 = off)")
	f.bass = fs.Float64("bass-db", 0, "bass shelf at 150 Hz, −12…+12 dB")
	f.presence = fs.Float64("presence-db", 1.5, "presence bell at 3.2 kHz, −12…+12 dB")
	f.treble = fs.Float64("treble-db", 0, "treble shelf at 4 kHz, −12…+12 dB")
	f.speechFloor = fs.Float64("speech-floor-db", -1, "noise-reduction limit inside speech, 0–100 dB (keeps quiet detail in words); default depends on the profile")
	f.toneMatch = fs.Float64("tone-match", -1, "automatic tonal correction towards a studio curve, 0–100 (0 = off); default depends on the profile")
	f.clarity = fs.Float64("clarity", -1, "more present, open voice (speech-gated presence and top end), 0–100 (0 = off); default depends on the profile")
	fs.Var(&f.eq, "eq", "extra EQ band `kind:hz:gain[:q]` (kind: low, high, peak), e.g. low:200:+9; repeatable")
	f.noDeess = fs.Bool("no-deess", false, "do not tame sibilance")
	f.noCompress = fs.Bool("no-compress", false, "do not even out the level of loud and soft words")
	f.noNormalize = fs.Bool("no-normalize", false, "files: keep the loudness, only guard against clipping; live: no automatic level")
	f.ceiling = fs.Float64("ceiling-db", -1, "peak ceiling in dB")
	f.engine = fs.String("engine", "native", "denoising engine: native (built in) or deepfilter (neural, needs DeepFilterNet installed separately)")
	f.dfBin = fs.String("deepfilter-bin", "", "files: path to the deep-filter tool (default: PATH, KYONE_DEEP_FILTER, next to kyone)")
	f.dfLib = fs.String("deepfilter-lib", "", "live: path to the DeepFilterNet C library libdf (default: KYONE_DF_LIB, next to kyone)")
	f.dfModel = fs.String("deepfilter-model", "", "DeepFilterNet model .tar.gz (live default: DeepFilterNet3_ll_onnx.tar.gz next to kyone; files: the tool's built-in model)")
	f.dfPostFilter = fs.Bool("deepfilter-postfilter", false, "DeepFilterNet post-filter: stronger noise removal, slightly more artefacts")
	return f
}

// options builds the enhancement options from the parsed flags, starting from
// the chosen profile and applying only the flags that were actually given.
func (f *enhanceFlags) options(fs *flag.FlagSet) (enhance.Options, error) {
	set := map[string]bool{}
	fs.Visit(func(fl *flag.Flag) { set[fl.Name] = true })
	if set["profile"] && set["attenuation-limit-db"] {
		return enhance.Options{}, errors.New("--profile and --attenuation-limit-db cannot be used together")
	}
	opt, err := enhance.OptionsFor(*f.profile)
	if err != nil {
		return opt, err
	}
	if set["attenuation-limit-db"] {
		opt.AttenuationLimitDB = *f.limit
		opt.Overshoot = enhance.OvershootFor(*f.limit)
	}
	if set["dereverb"] {
		opt.Dereverb = *f.dereverb
	}
	if set["clarity"] {
		opt.Clarity = *f.clarity
	}
	if set["tone-match"] {
		opt.ToneMatch = *f.toneMatch
	}
	if set["speech-floor-db"] {
		opt.SpeechFloorDB = *f.speechFloor
	}
	for _, spec := range f.eq {
		band, err := enhance.ParseEQBand(spec)
		if err != nil {
			return opt, err
		}
		opt.EQ = append(opt.EQ, band)
	}
	opt.HighPassHz = *f.highpass
	opt.BassDB, opt.PresenceDB, opt.TrebleDB = *f.bass, *f.presence, *f.treble
	opt.DeEss = !*f.noDeess
	opt.Compress = !*f.noCompress
	opt.Normalize = !*f.noNormalize
	opt.CeilingDB = *f.ceiling
	return opt, nil
}

// eqFlags collects repeated --eq values.
type eqFlags []string

func (e *eqFlags) String() string     { return strings.Join(*e, ",") }
func (e *eqFlags) Set(v string) error { *e = append(*e, v); return nil }

// neural returns the external denoiser the flags ask for, or nil for the
// native engine. For live use it needs the C library and a model, which are
// loaded now so that a missing file is reported before any audio device is
// opened; for files it needs the deep-filter tool.
func (f *enhanceFlags) neural(live bool) (*enhance.Neural, error) {
	switch *f.engine {
	case "native":
		return nil, nil
	case "deepfilter":
	default:
		return nil, fmt.Errorf("unknown engine %q (use native or deepfilter)", *f.engine)
	}
	n := &enhance.Neural{Name: "DeepFilterNet"}
	if !live {
		bin, err := deepfilter.FindCLI(*f.dfBin)
		if err != nil {
			return nil, err
		}
		model := *f.dfModel
		if model == "" {
			model = os.Getenv("KYONE_DF_MODEL")
		}
		cli := &deepfilter.CLI{Bin: bin, Model: model, PostFilter: *f.dfPostFilter}
		n.File = cli.Denoise
		return n, nil
	}
	libPath, err := deepfilter.FindLib(*f.dfLib)
	if err != nil {
		return nil, err
	}
	model, err := deepfilter.FindModel(*f.dfModel)
	if err != nil {
		return nil, err
	}
	lib, err := deepfilter.OpenLib(libPath)
	if err != nil {
		return nil, err
	}
	beta := 0.0
	if *f.dfPostFilter {
		beta = 0.02
	}
	n.Open = func(attenDB float64) (enhance.StreamDenoiser, error) {
		s, err := lib.NewStream(model, attenDB, beta)
		if err != nil {
			return nil, err
		}
		return streamAdapter{s}, nil
	}
	return n, nil
}

// streamAdapter lets a deepfilter.Stream satisfy enhance.StreamDenoiser (whose
// Process returns nothing).
type streamAdapter struct{ *deepfilter.Stream }

func (a streamAdapter) Process(in, out []float32) { a.Stream.Process(in, out) }
