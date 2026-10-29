package device

import (
	"strings"
	"testing"
)

func TestVirtualCableDetection(t *testing.T) {
	for name, want := range map[string]bool{
		"BlackHole 2ch":                                 true,
		"CABLE Input (VB-Audio Virtual Cable)":          true,
		"Voicemeeter Input (VB-Audio Voicemeeter VAIO)": true,
		"Haut-parleurs MacBook Pro":                     false,
		"Speakers (Realtek(R) Audio)":                   false,
		"AirPods Pro":                                   false,
	} {
		if got := IsVirtualCable(name); got != want {
			t.Errorf("IsVirtualCable(%q) = %v, want %v", name, got, want)
		}
	}
	devs := []Info{{Name: "Speakers"}, {Name: "BlackHole 16ch"}, {Name: "CABLE Input"}}
	if got := VirtualCable(devs); got != "BlackHole 16ch" {
		t.Errorf("VirtualCable = %q", got)
	}
	if VirtualCable([]Info{{Name: "Speakers"}}) != "" {
		t.Error("found a cable where there is none")
	}
	if InstallHint() == "" {
		t.Error("no install hint")
	}
}

func TestMatchSelectsByNameSubstring(t *testing.T) {
	devs := []Info{{Name: "AirPods Pro de Antoine"}, {Name: "Micro MacBook Pro"}, {Name: "Microsoft Teams Audio"}}
	if i, err := match(devs, "", Capture); err != nil || i != -1 {
		t.Errorf("empty selects the default: %d, %v", i, err)
	}
	if i, err := match(devs, "macbook", Capture); err != nil || i != 1 {
		t.Errorf("case-insensitive match: %d, %v", i, err)
	}
	_, err := match(devs, "nope", Playback)
	if err == nil || !strings.Contains(err.Error(), "Micro MacBook Pro") || !strings.Contains(err.Error(), "playback") {
		t.Errorf("error should list the devices: %v", err)
	}
}

func TestRingDropsOldestAndKeepsOrder(t *testing.T) {
	r := newRing(8)
	r.push([]float32{1, 2, 3, 4, 5, 6})
	if d := r.push([]float32{7, 8, 9, 10}); d != 2 {
		t.Fatalf("dropped %d, want 2", d)
	}
	got := make([]float32, 8)
	if n := r.pop(got); n != 8 {
		t.Fatalf("popped %d", n)
	}
	for i, want := range []float32{3, 4, 5, 6, 7, 8, 9, 10} {
		if got[i] != want {
			t.Fatalf("got %v", got)
		}
	}
	if n := r.pop(got); n != 0 {
		t.Errorf("empty ring returned %d samples", n)
	}

	r.push([]float32{1, 2, 3, 4, 5})
	if n := r.dropOldest(2); n != 3 || r.len() != 2 {
		t.Errorf("dropOldest removed %d, left %d", n, r.len())
	}
	two := make([]float32, 2)
	r.pop(two)
	if two[0] != 4 || two[1] != 5 {
		t.Errorf("kept %v, want the newest two", two)
	}
}

func TestPeakMeterTracksTheMaximumAndResets(t *testing.T) {
	var p peakMeter
	p.observe([]float32{0.1, -0.7, 0.3})
	p.observe([]float32{0.2, 0.5})
	if got := p.take(); got != 0.7 {
		t.Errorf("peak %v, want 0.7", got)
	}
	if got := p.take(); got != 0 {
		t.Errorf("peak after reset %v", got)
	}
}
