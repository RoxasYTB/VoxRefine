// Command match compares candidate recordings with a reference that was
// produced from the same input by another tool (for instance Adobe Podcast
// Enhance), to tune kyone's settings towards it. It is a development helper.
//
// For every file it prints the level in speech and in pauses per third-octave
// band, and a few scalar scores against the reference (level, log-spectral
// distance, SI-SDR after alignment). All files are converted to 48 kHz mono.
//
// With -fit it also searches for the shelf/bell EQ that brings the first
// candidate's speech spectrum closest to the reference, and prints it as
// kyone --eq flags.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"sort"
	"strings"

	"kyone/internal/audio"
	"kyone/internal/dsp"
)

const (
	rate   = 48000
	fftN   = 2048
	hopN   = 1024
	frameS = 0.05 // activity frames, seconds
)

func main() {
	bands := flag.Bool("bands", true, "print the per-band table")
	frames := flag.Bool("frames", false, "list the pause frames with input/reference/first-candidate levels (total and above 4 kHz)")
	fit := flag.Bool("fit", false, "search the EQ that maps the first candidate's speech spectrum onto the reference")
	flag.Parse()
	if flag.NArg() < 3 {
		fmt.Fprintln(os.Stderr, "usage: match [-bands=false] [-fit] input.wav reference.wav candidate.wav [...]")
		os.Exit(2)
	}
	files := flag.Args()
	sigs := make([][]float32, len(files))
	for i, p := range files {
		s, err := load(p)
		if err != nil {
			fmt.Fprintln(os.Stderr, "match:", err)
			os.Exit(1)
		}
		sigs[i] = s
	}
	in, ref := sigs[0], sigs[1]

	lag := bestLag(in, ref, rate/10)
	fmt.Printf("alignment: reference lags the input by %d samples (%.2f ms)\n\n", lag, 1000*float64(lag)/rate)

	// Speech / pause masks from the reference's own 50 ms RMS.
	speech, pause := masks(ref)

	type row struct {
		name           string
		sp, pa         []float64
		lsdSp, lsdPa   float64
		melSp, melPa   float64
		sisdr, levelDB float64
	}
	refSp, refPa := bandLevels(ref, speech), bandLevels(ref, pause)
	var rows []row
	for i, s := range sigs {
		n := min(len(s), len(ref))
		r := row{name: short(files[i]), sp: bandLevels(s[:n], speech), pa: bandLevels(s[:n], pause)}
		r.lsdSp, r.lsdPa = rmsDiff(r.sp, refSp), rmsDiff(r.pa, refPa)
		r.levelDB = 10 * math.Log10(power(s[:n], speech)/power(ref[:n], speech))
		r.melSp, r.melPa = melLSD(s[:n], ref[:n], speech), melLSD(s[:n], ref[:n], pause)
		if i >= 2 {
			r.sisdr = sisdr(s[:n], ref[:n], speech)
		}
		rows = append(rows, r)
	}

	if *bands {
		fmt.Printf("%-9s", "band Hz")
		for _, r := range rows {
			fmt.Printf(" | %-15s", r.name)
		}
		fmt.Printf("\n%-9s", "")
		for range rows {
			fmt.Printf(" | %7s %7s", "speech", "pause")
		}
		fmt.Println()
		for b, fc := range centres {
			fmt.Printf("%-9.0f", fc)
			for _, r := range rows {
				fmt.Printf(" | %7.1f %7.1f", r.sp[b], r.pa[b])
			}
			fmt.Println()
		}
		fmt.Println()
	}
	fmt.Printf("%-24s %9s %11s %11s %11s %11s\n", "file", "level dB", "band speech", "band pause", "mel speech", "mel pause")
	for _, r := range rows {
		fmt.Printf("%-24s %+9.1f %11.1f %11.1f %11.1f %11.1f\n", r.name, r.levelDB, r.lsdSp, r.lsdPa, r.melSp, r.melPa)
	}
	if *frames {
		bandDynamics(ref, sigs[2], speech)
	}
	fmt.Printf("\n%-24s %12s %12s\n", "file", "roughness sp", "roughness pa")
	for i, sg := range sigs {
		n := min(len(sg), len(ref))
		fmt.Printf("%-24s %9.1f dB %9.1f dB\n", rows[i].name, melFlux(sg[:n], speech), melFlux(sg[:n], pause))
	}
	fmt.Println("(roughness: mean frame-to-frame change of the mel spectrogram; higher = grainier / more fluctuating)")
	fmt.Printf("\n%-24s %18s %20s\n", "file", "harmonic contrast", "mel error by region")
	for i, sg := range sigs {
		n := min(len(sg), len(ref))
		lo, mid, hi := melRegions(sg[:n], ref[:n], speech)
		fmt.Printf("%-24s %15.1f dB %8.1f /%5.1f /%5.1f dB\n", rows[i].name, harmonicContrast(sg[:n], speech), lo, mid, hi)
	}
	fmt.Println("(contrast: spread of 300–2000 Hz bin levels inside speech, higher = crisper harmonics; regions: <500 Hz / 0.5–4 kHz / >4 kHz)")
	fmt.Println("\nlevel = speech RMS relative to the reference. band = RMS difference of long-term third-octave levels;")
	fmt.Println("mel = RMS difference of 40-band mel spectrograms frame by frame (dB, phase-insensitive). Lower is closer.")
	if *fit {
		fitEQ(rows[2].sp, refSp)
	}
	if *frames {
		listPauseFrames(in, ref, sigs[2], pause)
	}
}

func load(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := audio.ReadWAV(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return audio.Resample(b.ToChannels(1), rate).Ch[0], nil
}

func short(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			p = p[i+1:]
			break
		}
	}
	if len([]rune(p)) > 15 {
		r := []rune(p)
		return string(r[:14]) + "…"
	}
	return p
}

// masks marks the speech-dominated (top 45 % by level) and pause (bottom 15 %)
// 50 ms frames of x, returned as per-sample booleans.
func masks(x []float32) (speech, pause []bool) {
	win := int(frameS * rate)
	n := len(x) / win
	lv := make([]float64, n)
	for i := range lv {
		var e float64
		for _, v := range x[i*win : (i+1)*win] {
			e += float64(v) * float64(v)
		}
		lv[i] = e / float64(win)
	}
	sorted := append([]float64(nil), lv...)
	sort.Float64s(sorted)
	lo, hi := sorted[int(0.15*float64(n))], sorted[int(0.55*float64(n))]
	speech, pause = make([]bool, len(x)), make([]bool, len(x))
	for i, l := range lv {
		for j := i * win; j < (i+1)*win; j++ {
			speech[j] = l >= hi
			pause[j] = l <= lo
		}
	}
	return
}

// centres are the third-octave band centre frequencies used in the report.
var centres = func() []float64 {
	var c []float64
	for f := 100.0; f < 20000; f *= math.Pow(2, 1.0/3) {
		c = append(c, f)
	}
	return c
}()

// bandLevels is the average power spectrum of the frames selected by sel,
// summed into third-octave bands, in dB.
func bandLevels(x []float32, sel []bool) []float64 {
	fft := dsp.NewRFFT(fftN)
	buf := make([]float64, fftN)
	spec := make([]complex128, fft.Bins())
	acc := make([]float64, fft.Bins())
	var frames int
	for s := 0; s+fftN <= len(x); s += hopN {
		// Keep a frame only if its centre is selected.
		if !sel[s+fftN/2] {
			continue
		}
		for i := range buf {
			buf[i] = float64(x[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/fftN))
		}
		fft.Forward(buf, spec)
		for k, v := range spec {
			acc[k] += real(v)*real(v) + imag(v)*imag(v)
		}
		frames++
	}
	out := make([]float64, len(centres))
	for b, fc := range centres {
		lo, hi := fc/math.Pow(2, 1.0/6), fc*math.Pow(2, 1.0/6)
		var e float64
		for k := range acc {
			f := float64(k) * rate / fftN
			if f >= lo && f < hi {
				e += acc[k]
			}
		}
		out[b] = 10 * math.Log10(e/float64(max(frames, 1))/float64(fftN)/float64(fftN)+1e-20)
	}
	return out
}

func rmsDiff(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s / float64(len(a)))
}

func power(x []float32, sel []bool) float64 {
	var s float64
	var n int
	for i, v := range x {
		if sel[i] {
			s += float64(v) * float64(v)
			n++
		}
	}
	return s/float64(max(n, 1)) + 1e-20
}

func sisdr(x, ref []float32, sel []bool) float64 {
	var xr, rr float64
	for i := range x {
		if sel[i] {
			xr += float64(x[i]) * float64(ref[i])
			rr += float64(ref[i]) * float64(ref[i])
		}
	}
	a := xr / rr
	var tgt, e float64
	for i := range x {
		if sel[i] {
			t := a * float64(ref[i])
			d := float64(x[i]) - t
			tgt += t * t
			e += d * d
		}
	}
	return 10 * math.Log10(tgt/e)
}

// bestLag finds the shift d in [-max, max] maximising the correlation of
// ref[i+d] with in[i] (positive d: the reference is delayed).
func bestLag(in, ref []float32, max int) int {
	n := 1
	for n < len(in)+len(ref) {
		n <<= 1
	}
	fft := dsp.NewFFT(n)
	a, b := make([]complex128, n), make([]complex128, n)
	for i, v := range in {
		a[i] = complex(float64(v), 0)
	}
	for i, v := range ref {
		b[i] = complex(float64(v), 0)
	}
	fft.Forward(a)
	fft.Forward(b)
	for i := range a {
		a[i] = b[i] * complex(real(a[i]), -imag(a[i])) // ref · conj(in)
	}
	fft.Inverse(a)
	best, bestV := 0, math.Inf(-1)
	for d := -max; d <= max; d++ {
		if v := real(a[(d+n)%n]); v > bestV {
			best, bestV = d, v
		}
	}
	return best
}

// melLSD is the RMS difference, in dB, between the mel spectrograms of x and
// ref over the frames whose centre is selected by sel.
func melLSD(x, ref []float32, sel []bool) float64 {
	const (
		n     = 1024
		hop   = 512
		nMel  = 40
		floor = -110.0
	)
	fb := melFilterbank(n, nMel, 80, 16000)
	fft := dsp.NewRFFT(n)
	buf := make([]float64, n)
	spec := make([]complex128, fft.Bins())
	mel := func(sig []float32, s int) []float64 {
		for i := range buf {
			buf[i] = float64(sig[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/n))
		}
		fft.Forward(buf, spec)
		out := make([]float64, nMel)
		for m, w := range fb {
			var e float64
			for k, wk := range w {
				if wk != 0 {
					e += wk * (real(spec[k])*real(spec[k]) + imag(spec[k])*imag(spec[k]))
				}
			}
			out[m] = math.Max(10*math.Log10(e/float64(n)/float64(n)+1e-20), floor)
		}
		return out
	}
	var sum float64
	var count int
	for s := 0; s+n <= len(x); s += hop {
		if !sel[s+n/2] {
			continue
		}
		a, b := mel(x, s), mel(ref, s)
		for m := range a {
			d := a[m] - b[m]
			sum += d * d
			count++
		}
	}
	if count == 0 {
		return math.NaN()
	}
	return math.Sqrt(sum / float64(count))
}

// melFilterbank returns nMel triangular filters over the n/2+1 FFT bins.
func melFilterbank(n, nMel int, lo, hi float64) [][]float64 {
	mel := func(f float64) float64 { return 2595 * math.Log10(1+f/700) }
	inv := func(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }
	pts := make([]float64, nMel+2)
	for i := range pts {
		pts[i] = inv(mel(lo) + (mel(hi)-mel(lo))*float64(i)/float64(nMel+1))
	}
	fb := make([][]float64, nMel)
	for m := range fb {
		fb[m] = make([]float64, n/2+1)
		for k := range fb[m] {
			f := float64(k) * rate / float64(n)
			switch {
			case f > pts[m] && f <= pts[m+1]:
				fb[m][k] = (f - pts[m]) / (pts[m+1] - pts[m])
			case f > pts[m+1] && f < pts[m+2]:
				fb[m][k] = (pts[m+2] - f) / (pts[m+2] - pts[m+1])
			}
		}
	}
	return fb
}

// fitEQ finds the low shelf, high shelf and peaking bell whose combined
// response best matches ref − cand (speech band levels, mean removed) and
// prints the result as kyone flags.
func fitEQ(cand, ref []float64) {
	d := make([]float64, len(centres))
	w := make([]float64, len(centres))
	for b := range d {
		d[b] = ref[b] - cand[b]
		w[b] = 1
		if centres[b] > 14000 {
			w[b] = 0.3
		}
	}
	type option struct {
		label string
		resp  []float64
		cost  float64 // regularisation
	}
	respOf := func(c dsp.Cascade) []float64 {
		r := make([]float64, len(centres))
		for b, fc := range centres {
			r[b] = dsp.DB(cmplx.Abs(c.Response(fc, rate)))
		}
		return r
	}
	none := option{label: "", resp: make([]float64, len(centres))}
	lows, highs, peaks := []option{none}, []option{none}, []option{none}
	for _, fc := range []float64{80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800} {
		for g := -12.0; g <= 18; g += 1.5 {
			if g != 0 {
				lows = append(lows, option{fmt.Sprintf("--eq low:%g:%+g", fc, g), respOf(dsp.Cascade{dsp.LowShelf(rate, fc, g)}), g * g})
			}
		}
	}
	for _, fc := range []float64{500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000} {
		for g := -18.0; g <= 9; g += 1.5 {
			if g != 0 {
				highs = append(highs, option{fmt.Sprintf("--eq high:%g:%+g", fc, g), respOf(dsp.Cascade{dsp.HighShelf(rate, fc, g)}), g * g})
			}
		}
	}
	for _, fc := range centres {
		if fc < 160 || fc > 9000 {
			continue
		}
		for _, q := range []float64{0.7, 1.4, 2.8} {
			for g := -9.0; g <= 9; g += 1.5 {
				if g != 0 {
					peaks = append(peaks, option{fmt.Sprintf("--eq peak:%.0f:%+g:%g", fc, g, q), respOf(dsp.Cascade{dsp.Peaking(rate, fc, q, g)}), g * g})
				}
			}
		}
	}

	const reg = 0.002 // prefers gentler EQs when the fit is equally good
	best := math.Inf(1)
	var bl, bh, bp option
	sum := make([]float64, len(centres))
	for _, l := range lows {
		for _, h := range highs {
			for b := range sum {
				sum[b] = l.resp[b] + h.resp[b]
			}
			for _, p := range peaks {
				var wsum, mean float64
				for b := range sum {
					mean += w[b] * (d[b] - sum[b] - p.resp[b])
					wsum += w[b]
				}
				mean /= wsum
				var cost float64
				for b := range sum {
					r := d[b] - sum[b] - p.resp[b] - mean
					cost += w[b] * r * r
				}
				cost = cost/wsum + reg*(l.cost+h.cost+p.cost)/float64(len(centres))
				if cost < best {
					best, bl, bh, bp = cost, l, h, p
				}
			}
		}
	}
	// Residual of the best fit, band by band.
	var mean, wsum float64
	for b := range d {
		mean += w[b] * (d[b] - bl.resp[b] - bh.resp[b] - bp.resp[b])
		wsum += w[b]
	}
	mean /= wsum
	var res float64
	for b := range d {
		r := d[b] - bl.resp[b] - bh.resp[b] - bp.resp[b] - mean
		res += w[b] * r * r
	}
	var flags []string
	for _, o := range []option{bl, bh, bp} {
		if o.label != "" {
			flags = append(flags, o.label)
		}
	}
	fmt.Printf("\nbest EQ fit (speech, third-octave): rms residual %.2f dB (before: %.2f dB)\n  %s\n  (the remaining %+.1f dB overall offset is a level difference, handled by --target-lufs)\n",
		math.Sqrt(res/wsum), rmsOf(d, w), strings.Join(flags, " "), mean)
}

func rmsOf(d, w []float64) float64 {
	var mean, wsum float64
	for b := range d {
		mean += w[b] * d[b]
		wsum += w[b]
	}
	mean /= wsum
	var s float64
	for b := range d {
		s += w[b] * (d[b] - mean) * (d[b] - mean)
	}
	return math.Sqrt(s / wsum)
}

// listPauseFrames prints, for each 50 ms pause frame, the level of the input,
// the reference and the first candidate, in total and above 4 kHz (dBFS).
func listPauseFrames(in, ref, cand []float32, pause []bool) {
	win := int(frameS * rate)
	level := func(x []float32, s int) (total, hf float64) {
		fft := dsp.NewRFFT(2048)
		buf := make([]float64, 2048)
		spec := make([]complex128, fft.Bins())
		for i := range buf {
			buf[i] = float64(x[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/2048))
		}
		fft.Forward(buf, spec)
		var t, h float64
		for k, v := range spec {
			e := real(v)*real(v) + imag(v)*imag(v)
			t += e
			if float64(k)*rate/2048 >= 4000 {
				h += e
			}
		}
		norm := 2048.0 * 2048.0
		return 10 * math.Log10(t/norm+1e-20), 10 * math.Log10(h/norm+1e-20)
	}
	fmt.Printf("\n%7s | %16s | %16s | %16s\n%7s | %7s %8s | %7s %8s | %7s %8s\n", "t (s)", "input", "reference", "candidate", "", "total", ">4kHz", "total", ">4kHz", "total", ">4kHz")
	n := min(len(in), len(ref), len(cand))
	for s := 0; s+2048 <= n && s/win*win+win <= n; s += win {
		if !pause[s+win/2] {
			continue
		}
		it, ih := level(in, s)
		rt, rh := level(ref, s)
		ct, ch := level(cand, s)
		fmt.Printf("%7.2f | %7.1f %8.1f | %7.1f %8.1f | %7.1f %8.1f\n", float64(s)/rate, it, ih, rt, rh, ct, ch)
	}
}

// harmonicContrast is the mean, over speech frames, of the spread (90th minus
// 10th percentile) of the 300–2000 Hz bin levels: crisp harmonics over clean
// valleys give a high value, smeared or noisy ones a low value.
func harmonicContrast(x []float32, sel []bool) float64 {
	const n, hop = 2048, 1024
	fft := dsp.NewRFFT(n)
	buf := make([]float64, n)
	spec := make([]complex128, fft.Bins())
	var sum float64
	var count int
	for s := 0; s+n <= len(x); s += hop {
		if !sel[s+n/2] {
			continue
		}
		for i := range buf {
			buf[i] = float64(x[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/n))
		}
		fft.Forward(buf, spec)
		var lv []float64
		for k := int(300 * n / rate); k <= int(2000*n/rate); k++ {
			lv = append(lv, 10*math.Log10(real(spec[k])*real(spec[k])+imag(spec[k])*imag(spec[k])+1e-12))
		}
		sort.Float64s(lv)
		sum += lv[len(lv)*9/10] - lv[len(lv)/10]
		count++
	}
	return sum / float64(max(count, 1))
}

// melRegions splits the mel-spectrogram error over speech frames into three
// frequency regions (below 500 Hz, 0.5–4 kHz, above 4 kHz).
func melRegions(x, ref []float32, sel []bool) (lo, mid, hi float64) {
	const (
		n    = 1024
		hop  = 512
		nMel = 40
	)
	fb := melFilterbank(n, nMel, 80, 16000)
	centre := make([]float64, nMel)
	for m, w := range fb {
		var num, den float64
		for k, wk := range w {
			num += wk * float64(k) * rate / n
			den += wk
		}
		centre[m] = num / den
	}
	fft := dsp.NewRFFT(n)
	buf := make([]float64, n)
	spec := make([]complex128, fft.Bins())
	mel := func(sig []float32, s int) []float64 {
		for i := range buf {
			buf[i] = float64(sig[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/n))
		}
		fft.Forward(buf, spec)
		out := make([]float64, nMel)
		for m, w := range fb {
			var e float64
			for k, wk := range w {
				e += wk * (real(spec[k])*real(spec[k]) + imag(spec[k])*imag(spec[k]))
			}
			out[m] = math.Max(10*math.Log10(e/float64(n)/float64(n)+1e-20), -110)
		}
		return out
	}
	var sums, counts [3]float64
	for s := 0; s+n <= len(x); s += hop {
		if !sel[s+n/2] {
			continue
		}
		a, b := mel(x, s), mel(ref, s)
		for m := range a {
			r := 1
			if centre[m] < 500 {
				r = 0
			} else if centre[m] > 4000 {
				r = 2
			}
			d := a[m] - b[m]
			sums[r] += d * d
			counts[r]++
		}
	}
	f := func(i int) float64 { return math.Sqrt(sums[i] / math.Max(counts[i], 1)) }
	return f(0), f(1), f(2)
}

// melFlux is the mean absolute frame-to-frame change (dB) of the 40-band mel
// spectrogram over frames inside the selection: a measure of how grainy or
// fluctuating the signal is, whatever its level.
func melFlux(x []float32, sel []bool) float64 {
	const (
		n    = 1024
		hop  = 256
		nMel = 40
	)
	fb := melFilterbank(n, nMel, 80, 16000)
	fft := dsp.NewRFFT(n)
	buf := make([]float64, n)
	spec := make([]complex128, fft.Bins())
	var prev []float64
	var sum float64
	var count int
	for s := 0; s+n <= len(x); s += hop {
		for i := range buf {
			buf[i] = float64(x[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/n))
		}
		fft.Forward(buf, spec)
		cur := make([]float64, nMel)
		for m, w := range fb {
			var e float64
			for k, wk := range w {
				e += wk * (real(spec[k])*real(spec[k]) + imag(spec[k])*imag(spec[k]))
			}
			cur[m] = math.Max(10*math.Log10(e/float64(n)/float64(n)+1e-20), -110)
		}
		if prev != nil && sel[s+n/2] && sel[s-hop+n/2] {
			for m := range cur {
				sum += math.Abs(cur[m] - prev[m])
				count++
			}
		}
		prev = cur
	}
	return sum / float64(max(count, 1))
}

// bandDynamics regresses, per frequency region, the reference's frame levels
// on the first candidate's over the speech frames. A slope below 1 means the
// reference compresses that region's dynamics more than the candidate does.
func bandDynamics(ref, cand []float32, sel []bool) {
	const n, hop = 2048, 1024
	regions := []struct {
		name   string
		lo, hi float64
	}{{"<250 Hz", 60, 250}, {"250-1k", 250, 1000}, {"1-4k", 1000, 4000}, {">4k", 4000, 16000}}
	fft := dsp.NewRFFT(n)
	buf := make([]float64, n)
	spec := make([]complex128, fft.Bins())
	level := func(x []float32, s int) []float64 {
		for i := range buf {
			buf[i] = float64(x[s+i]) * (0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/n))
		}
		fft.Forward(buf, spec)
		out := make([]float64, len(regions))
		for r, rg := range regions {
			var e float64
			for k, v := range spec {
				if f := float64(k) * rate / n; f >= rg.lo && f < rg.hi {
					e += real(v)*real(v) + imag(v)*imag(v)
				}
			}
			out[r] = 10 * math.Log10(e/float64(n)/float64(n)+1e-20)
		}
		return out
	}
	xs := make([][]float64, len(regions))
	ys := make([][]float64, len(regions))
	m := min(len(ref), len(cand))
	for s := 0; s+n <= m; s += hop {
		if !sel[s+n/2] {
			continue
		}
		c, r := level(cand, s), level(ref, s)
		for i := range regions {
			xs[i] = append(xs[i], c[i])
			ys[i] = append(ys[i], r[i])
		}
	}
	fmt.Printf("\nreference vs candidate frame levels over speech (y = slope·x + offset):\n")
	for i, rg := range regions {
		var mx, my float64
		for j := range xs[i] {
			mx += xs[i][j]
			my += ys[i][j]
		}
		nn := float64(len(xs[i]))
		mx, my = mx/nn, my/nn
		var sxx, sxy, syy float64
		for j := range xs[i] {
			dx, dy := xs[i][j]-mx, ys[i][j]-my
			sxx += dx * dx
			sxy += dx * dy
			syy += dy * dy
		}
		slope := sxy / sxx
		fmt.Printf("  %-8s slope %.2f  correlation %.2f  spread cand %.1f dB / ref %.1f dB\n", rg.name, slope, sxy/math.Sqrt(sxx*syy), math.Sqrt(sxx/nn), math.Sqrt(syy/nn))
	}
}
