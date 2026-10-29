// Package enhance turns a raw voice recording into a clean, even, broadcast-
// level one: noise and room reduction, tonal polish, de-essing, levelling and
// loudness normalisation.
package enhance

import (
	"fmt"
	"strings"
)

// SampleRate is the rate the pipeline works at and writes.
const SampleRate = 48000

// Options configures the whole chain. Zero values disable the matching stage,
// except where noted.
type Options struct {
	// AttenuationLimitDB caps noise reduction (0–100 dB). 0 disables denoising.
	AttenuationLimitDB float64
	// SpeechFloorDB is the shallower noise-reduction limit applied while
	// speech is present, so quiet detail inside words is kept. 0 disables it.
	SpeechFloorDB float64
	// Overshoot (>= 1) scales the noise estimate: higher removes more noise.
	Overshoot float64
	// Dereverb is the room-reverb suppression strength, 0–100. 0 disables it.
	Dereverb float64

	// ToneMatch (0–100) steers the voice's tonal balance towards a studio
	// curve, correcting thin or boomy microphones automatically. 0 disables it.
	ToneMatch float64

	// Clarity (0–100) makes the voice more present and open: a speech-gated
	// presence lift and regenerated top end. 0 disables it.
	Clarity float64

	// HighPassHz removes rumble below this frequency. 0 disables it.
	HighPassHz float64
	// Tone shaping, in dB (−12…+12): low shelf at 150 Hz, presence bell at
	// 3.2 kHz, high shelf at 4 kHz.
	BassDB     float64
	PresenceDB float64
	TrebleDB   float64
	// EQ adds free-form bands after the three fixed tone controls.
	EQ []EQBand

	DeEss    bool // tame sibilance
	Compress bool // even out the level of loud and soft words

	// Normalize brings the programme to TargetLUFS, then guarantees that no
	// true peak exceeds CeilingDB.
	Normalize  bool
	TargetLUFS float64
	CeilingDB  float64

	// Mono mixes the result down to one channel.
	Mono bool

	// Neural, if set, replaces kyone's own spectral denoiser with an external
	// neural one (DeepFilterNet); AttenuationLimitDB still sets how far it may
	// reduce noise. Everything else in the chain stays.
	Neural *Neural
}

// EQBand is one equaliser band: a low shelf, high shelf or peaking bell.
type EQBand struct {
	Kind   string  // "low", "high" or "peak"
	Hz     float64 // corner (shelves) or centre (peak) frequency
	GainDB float64
	Q      float64 // peak only; 0 selects 0.8
}

// ParseEQBand parses "kind:hz:gain[:q]", e.g. "low:200:+9", "high:900:-6" or
// "peak:3000:2:1.2". Gains are in dB.
func ParseEQBand(spec string) (EQBand, error) {
	parts := strings.Split(spec, ":")
	if len(parts) < 3 || len(parts) > 4 {
		return EQBand{}, fmt.Errorf("invalid EQ band %q (want kind:hz:gain[:q], e.g. low:200:+9)", spec)
	}
	b := EQBand{Kind: parts[0]}
	if b.Kind != "low" && b.Kind != "high" && b.Kind != "peak" {
		return EQBand{}, fmt.Errorf("invalid EQ band %q: kind must be low, high or peak", spec)
	}
	nums := make([]float64, len(parts)-1)
	for i, p := range parts[1:] {
		if _, err := fmt.Sscanf(p, "%g", &nums[i]); err != nil {
			return EQBand{}, fmt.Errorf("invalid EQ band %q: %q is not a number", spec, p)
		}
	}
	b.Hz, b.GainDB = nums[0], nums[1]
	if len(nums) == 3 {
		if b.Kind != "peak" {
			return EQBand{}, fmt.Errorf("invalid EQ band %q: only peak bands take a Q", spec)
		}
		b.Q = nums[2]
	}
	return b, nil
}

// String renders the band in the syntax ParseEQBand accepts.
func (b EQBand) String() string {
	if b.Kind == "peak" && b.Q != 0 {
		return fmt.Sprintf("%s:%g:%+g:%g", b.Kind, b.Hz, b.GainDB, b.Q)
	}
	return fmt.Sprintf("%s:%g:%+g", b.Kind, b.Hz, b.GainDB)
}

// Profile is a named preset: how hard to denoise and how much automatic
// enhancement to apply.
type Profile struct {
	Name               string
	AttenuationLimitDB float64
	Overshoot          float64
	SpeechFloorDB      float64
	Dereverb           float64
	ToneMatch          float64
	Clarity            float64
	Description        string
}

var profiles = []Profile{
	{Name: "natural", AttenuationLimitDB: 12, Overshoot: 1.0, Dereverb: 15,
		Description: "gentle: keeps the room feel and the original tone, leaves some residual noise"},
	{Name: "balanced", AttenuationLimitDB: 24, Overshoot: 1.1, SpeechFloorDB: 8, Dereverb: 35, ToneMatch: 50, Clarity: 30,
		Description: "default: clear voice, quiet natural background, light automatic tone correction"},
	{Name: "strong", AttenuationLimitDB: 40, Overshoot: 1.2, SpeechFloorDB: 10, Dereverb: 60, ToneMatch: 70, Clarity: 40,
		Description: "aggressive: studio-dry voice, may sound slightly processed"},
	{Name: "studio", AttenuationLimitDB: 40, Overshoot: 1.2, SpeechFloorDB: 8, Dereverb: 20, ToneMatch: 100, Clarity: 60,
		Description: "full enhancement: deep noise removal, warm studio tone, present voice (closest to Adobe Enhance)"},
}

// DefaultProfile is used when none is requested.
const DefaultProfile = "balanced"

// Profiles lists the presets in order of strength.
func Profiles() []Profile { return append([]Profile(nil), profiles...) }

// ProfileNames returns the preset names, for help texts.
func ProfileNames() string {
	names := make([]string, len(profiles))
	for i, p := range profiles {
		names[i] = p.Name
	}
	return strings.Join(names, ", ")
}

// DefaultOptions returns the options of the default profile with the full
// chain enabled.
func DefaultOptions() Options {
	o, _ := OptionsFor(DefaultProfile)
	return o
}

// OptionsFor returns the full-chain options for a named profile.
func OptionsFor(name string) (Options, error) {
	for _, p := range profiles {
		if p.Name == name {
			return Options{
				AttenuationLimitDB: p.AttenuationLimitDB,
				Overshoot:          p.Overshoot,
				SpeechFloorDB:      p.SpeechFloorDB,
				Dereverb:           p.Dereverb,
				ToneMatch:          p.ToneMatch,
				Clarity:            p.Clarity,
				HighPassHz:         80,
				PresenceDB:         1.5,
				DeEss:              true,
				Compress:           true,
				Normalize:          true,
				TargetLUFS:         -16,
				CeilingDB:          -1,
			}, nil
		}
	}
	return Options{}, fmt.Errorf("unknown profile %q (choose from: %s)", name, ProfileNames())
}

// OvershootFor derives the over-subtraction factor from a custom attenuation
// limit by interpolating between the natural and strong presets.
func OvershootFor(limitDB float64) float64 {
	lo, hi := profiles[0], profiles[2] // natural … strong
	t := (limitDB - lo.AttenuationLimitDB) / (hi.AttenuationLimitDB - lo.AttenuationLimitDB)
	t = min(max(t, 0), 1)
	return lo.Overshoot + t*(hi.Overshoot-lo.Overshoot)
}

// Validate reports the first out-of-range setting.
func (o Options) Validate() error {
	check := func(name string, v, lo, hi float64) error {
		if v < lo || v > hi || v != v {
			return fmt.Errorf("%s must be between %g and %g (got %g)", name, lo, hi, v)
		}
		return nil
	}
	for _, b := range o.EQ {
		if err := b.validate(); err != nil {
			return err
		}
	}
	for _, c := range []struct {
		name      string
		v, lo, hi float64
	}{
		{"attenuation limit (dB)", o.AttenuationLimitDB, 0, 100},
		{"overshoot", o.Overshoot, 0, 4},
		{"dereverb", o.Dereverb, 0, 100},
		{"tone match", o.ToneMatch, 0, 100},
		{"clarity", o.Clarity, 0, 100},
		{"high-pass frequency (Hz)", o.HighPassHz, 0, 500},
		{"bass (dB)", o.BassDB, -12, 12},
		{"presence (dB)", o.PresenceDB, -12, 12},
		{"treble (dB)", o.TrebleDB, -12, 12},
		{"target loudness (LUFS)", o.TargetLUFS, -40, -5},
		{"ceiling (dB)", o.CeilingDB, -12, 0},
	} {
		if c.name == "target loudness (LUFS)" && !o.Normalize {
			continue
		}
		if err := check(c.name, c.v, c.lo, c.hi); err != nil {
			return err
		}
	}
	return nil
}

func (b EQBand) validate() error {
	if b.Hz < 20 || b.Hz > 20000 {
		return fmt.Errorf("EQ band %s: frequency must be between 20 and 20000 Hz", b)
	}
	if b.GainDB < -24 || b.GainDB > 24 {
		return fmt.Errorf("EQ band %s: gain must be between -24 and +24 dB", b)
	}
	if b.Q != 0 && (b.Q < 0.2 || b.Q > 10) {
		return fmt.Errorf("EQ band %s: Q must be between 0.2 and 10", b)
	}
	return nil
}
