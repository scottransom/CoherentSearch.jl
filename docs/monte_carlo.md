# These are notes for the design of the monte carlo script to test CoherentSearch.jl against riptide, accelsearch, and prepfold

This file describes more details and constraints for the monte carlo mentioned in §3.2 of Summary_and_Future_Work.md.

The goal at this stage is to measure detection efficiency of gaussian-like (let's use von Mises profiles) profiles that span realistic ranges of pulse duty cycle and pulse pulse period at "threshold-like" known S/N values.

The simulations should be driven by a python script which can save its state and have its results files combined with those from other runs (i.e. in parallel on other CPUs/GPUs) to improve statistics.

## Outline of the routine:

  1. Randomly choose theoretical S/N (snr), duty cycle (dcyl), spin frequency (f0).
  2. Generate white noise time series and add pulsed signal at appropriate level (determining analytic S/N needs to be figured out)
  3. Analyze the time series:
    - fold with `prepfold` from PRESTO. Determine "S/N" via "sigma" value in .bestprof file
    - Search with riptide's `rseek`
  4. FFT the time series using `realfft` from PRESTO
  5. Search the un-rednoise removed FFT with `accelsearch`
  6. Remove "rednoise" from FFT using `rednoise`
  7. Search the _red.fft using `coherent_search.jl`
  8. Save binary detection yes/no via method and record S/N.  For `accelsearch` we should use the single-trial significances determined from the detected and optimized harmonics as reported by `ACCEL_sift.py` or `quick_prune_cands.py` rather than the values in the ACCEL output files. The time taken for the various searches should also be recorded. For `coherent_search.jl` and `rseek` we also want to record the detected duty cycle.
  9. Repeat ad nauseum.

## Technical details of the search parameters

dt = 60 microseconds (sample duration)
N = 2**24 (number of samples in time series)
T ~ 1006 sec, given the above
Injected S/N values: quantized at [6, 7, 8, 9, 10, 11]
f0:  Two different ranges, 2/3 vs 1/3:
  - 0.2-50 Hz, uniformly(?) distributed ("slow" pulsars, 2/3 of the injected signals)
  - gaussian distribution with 3.5 ms mean, 1 ms stdev ("MSPs", 1/3 of the signals)
dcyl:
  - For slow pulsars: 0.1% - 20%  (probably log-normal with mean around 2-3%)
  - For MSPs: 2%-30% uniform distribution 

## Questions to decide:

 - Should we inject more than one signal in each time series? I think probably yes.  Maybe even up to 4-5.  That will help our efficiency, and with long enough time series, there should be no (or very few) collisions.
 - Should we allow detections at harmonics? Probably yes, but those should be flagged. So we probably want to check the simplest harmonic ratios for the known f0 values (2x, 3x, 1/2x, 1/3x, at least)
 - Should the slow pulsars really be uniformly distributed?  Or should they be log-normal?  Probably the latter.
 - Does it make sense to drive the monte carlo via Julia rather than Python in order to avoid the Julia startup costs?

## Technical details about the calling sequences for the routines:

 - `prepfold` should default to the number of bins for MSPs.  For all slow pulsars, use `-n 128`.  Always use `-fine -noxwin -nopdsearch`.
 - `accelsearch` should use `-numharm 16 -zmax 0`
 - `rseek` somehow needs to use good values for --bmin and --bmax given the large freq ranges?
 - `coherent_search.jl` should be run with the default settings

---

# Run 1 — 82,014 injections, 2026-08-28

**What was run.** `bla0`, 48 workers, `--deep-every 3 --noise-every 10`,
`mc/mc_simulate.py` at 216b210…7ca4463, overnight 01:34–08:33. The band-limited
boxcar normalisation (1b8fed9, 00:28) is *in* — this is the first study run on
the corrected statistic. It completed **15,212 realisations of the 200,000
requested** (1,543 injection-free), i.e. **82,014 injections**, before being
interrupted; `rseek_B` ran on 2,197 of them (11,862 injections). Zero
`coherent_search` failures in 15,212 runs. Results in `~/Downloads/mc_results_1`.

Median wall per realisation: `rseek_B` 121.1 s, `rseek_A` 29.2 s,
**`coherent` 20.2 s** (`-t 1`, including ~2 s of Julia start-up),
`prepfold`×6 5.8 s, generate 4.5 s, `accelsearch` 1.7 s, `realfft` 0.5 s,
`rednoise` 0.3 s.

## 1. The headline, corrected for the subset trap

`mc_analyze.py` and `mc_quicklook.py` compare `rseek_B` — which ran on the
1-in-3 `--deep-every` subset — against a `coherent` column computed from 7x more
data. The `*` footnote flags it but the tables still put the two side by side.
Restricted to the 2,197 realisations both codes saw, matched at 0.01 false alarms
per realisation:

| | accelsearch | rseek_A | rseek_B | **coherent** |
|---|---|---|---|---|
| threshold at FAP 1e-2 | 8.00 | 7.95 | 8.05 | **7.05** |
| detected | 28.8% | 41.8% | 64.3% | **71.6%** |
| median wall | 1.7 s | 29.2 s | 121.1 s | **20.2 s** |

Head-to-head on that subset: both 7,441, **coherent-only 1,054, rseek_B-only
181**. So we beat riptide's *deep tiling* — the configuration that costs 6x our
wall clock — by 7.3 points, and its matched-coverage configuration by 30.

By injected S/N (deep subset, matched FAP 1e-2):

| S/N | accelsearch | rseek_A | rseek_B | coherent | N |
|---|---|---|---|---|---|
| 6 | 0.8% | 4.4% | 8.3% | **15.3%** | 1985 |
| 7 | 4.1% | 14.9% | 29.7% | **43.6%** | 1914 |
| 8 | 12.6% | 32.7% | 62.9% | **77.2%** | 1969 |
| 9 | 32.0% | 51.7% | 87.0% | **93.8%** | 2051 |
| 10 | 52.3% | 67.5% | 96.5% | **98.8%** | 1978 |
| 11 | 70.3% | 78.6% | 99.5% | **99.6%** | 1965 |

By spin frequency (same cut) we lead in every band: 68.4 / 70.9 / 71.1 / 73.2 /
72.1 / 72.6 / 75.3% over 0.1–1 / 1–5 / 5–20 / 20–100 / 100–200 / 200–400 /
400–1000 Hz, essentially flat, against `rseek_A`'s strong rise with frequency
(19.3% → 74.0%) and `rseek_B`'s flat ~55–72%.

**The ranking is invariant in FAP** — this is worth a figure, because it means
the result does not depend on the one threshold choice:

| FAP/realisation | accelsearch | rseek_A | rseek_B | coherent |
|---|---|---|---|---|
| 1 | 53.8% | 57.1% | 78.7% | **83.6%** |
| 0.1 | 39.4% | 49.1% | 71.6% | **78.0%** |
| 0.01 | 28.8% | 41.8% | 64.3% | **71.6%** |
| 0.001 | 20.8% | 36.7% | 56.8% | **66.6%** |

Our margin *grows* as the FAP tightens (4.9 → 9.8 points over `rseek_B`).

## 2. Why we win — the advantage separates cleanly into two terms

Report these apart. They have different causes and different lessons.

**(a) Threshold, worth 0.90 against `rseek_A`.** The noise-only maximum over an
injection-free realisation:

| | mean | sd | p99 | max |
|---|---|---|---|---|
| accelsearch | 6.479 | 0.539 | 7.97 | 9.12 |
| rseek_A | 7.148 | 0.234 | 7.90 | 8.40 |
| rseek_B | 7.319 | 0.217 | 7.90 | 8.10 |
| **coherent** | **6.431** | **0.197** | **6.99** | **7.43** |

(n = 1,543 empty realisations; 220 for `rseek_B`. Our low tail is censored at the
reporting floor of 6.0, so read p99, not the sd.) Part of this is
harmonic-family collapsing — 98.9% of our detections are labelled `1` against
`rseek_A`'s 94.6% and `accelsearch`'s 93.6%, and we report a median of 12
candidates against `rseek_A`'s ~162 — but the bigger part is (b).

Our false-alarm tail is also **Gaussian to the digit**: 7.51 per realisation at
≥6.0, 0.38 at 6.5, 0.010 at 7.0. `exp(-(7²−6²)/2)·7.51 = 0.011`. `rseek`'s is
heavier than Gaussian by ~8x over the same span (162 → 1.68 → 0.007), which is
what forces its threshold up.

**(b) The statistic's own variance.** At injected S/N 11 and duty 4–16%
(n = 5,138, detection ~100%, so no selection bias):

| | median | sd | IQR |
|---|---|---|---|
| coherent | 10.46 | **1.005** | 1.35 |
| rseek_A | 10.50 | 1.574 | 2.20 |
| prepfold_snr1 | 10.17 | 1.909 | 2.38 |

**Ours is a unit-variance statistic to three digits.** That is `--sigma
analytic` (2026-08-24) paying off exactly as designed: the other codes estimate
σ̂ from each profile and carry 1.6–1.9x the variance. Estimated-σ̂ noise inflates
both the recovered S/N *and* the false-alarm tail, so at matched FAP the
inflation cancels and the extra variance is a pure loss. This also retires the
last of the "our S/N 12.30 vs riptide's 11.80 on PM0063" puzzle — riptide's
recovered/injected ratio here runs 1.05–1.09 at broad duty, i.e. systematically
above unity, which is max-selection bias on a noisy σ̂, not extra sensitivity.

**(c) Recovery, duty-dependent.** Paired median Δstatistic (coherent −
`rseek_A`, same injection, same noise realisation), and the effective margin
after adding the 0.90 threshold gap:

| duty | n | median Δ | effective margin |
|---|---|---|---|
| <1% | 1977 | +2.05 | +2.95 |
| 1–2% | 6845 | +1.62 | +2.52 |
| 2–4% | 11908 | +0.85 | +1.75 |
| 4–8% | 13431 | +0.17 | +1.07 |
| 8–16% | 13133 | **−0.62** | +0.28 |
| 16–50% | 16033 | −0.25 | +0.65 |

`rseek_A` recovers a *higher* statistic than us at 8–16% duty and still loses,
because of (a) and (b). Counterfactual check on the full set: at our own
threshold we detect 70.9%; at `rseek_A`'s 7.95 we detect 53.9%, against its
42.0% — so roughly half the advantage is threshold and half is recovery.

**No systematic hole.** Injections missed by us but seen by `prepfold` at
snr1 > 10: **15 of 82,014** (median f0 329 Hz, median injected S/N 6). Same
count for `rseek_A`: **147**.

## 3. The one place we lose: duty < 0.5%

Splitting the lowest bin (deep subset, matched FAP 1e-2):

| duty | N | coherent | rseek_B | coh-only | B-only |
|---|---|---|---|---|---|
| <0.5% | 118 | 50.8 ± 4.6% | **64.4 ± 4.4%** | 3 | 19 |
| 0.5–1% | 655 | **62.1 ± 1.9%** | 59.8 ± 1.9% | 31 | 16 |
| 1–2% | 1655 | **70.5 ± 1.1%** | 63.4 ± 1.2% | 134 | 17 |
| >2% | 9434 | **72.7 ± 0.5%** | 64.7 ± 0.5% | 886 | 129 |

**The gap is entirely below 0.5% duty, not below 1%** — the quick-look's
`--by ducy` edges start at 0.01 and hide the split. It is 1.4% of the drawn
population, 118 injections in the subset, ~3σ on discordant pairs. It is real
and it is under-sampled; run 2 should stratify for it.

**It is harmonic truncation, not the filter.** Median recovered/injected by duty:

| duty | coherent | rseek_A | rseek_B | prepfold_snr1 |
|---|---|---|---|---|
| <0.5% | 0.811 | 0.662 | **0.975** | 0.777 |
| 0.5–1% | 0.919 | 0.709 | 1.020 | 0.868 |
| 1–2% | 0.969 | 0.789 | 1.038 | 0.920 |
| 2–4% | 0.984 | 0.873 | 1.050 | 0.937 |
| 4–8% | 0.990 | 0.971 | 1.070 | 0.957 |
| 8–16% | 0.988 | 1.050 | 1.089 | 0.963 |
| 16–50% | 0.972 | 1.000 | 1.011 | 0.836 |

(Detected-only, so upward-biased for the low-detection cells; `rseek_B`'s
1.05–1.09 is the σ̂ inflation of §2b. `prepfold` detects everything so its column
is unbiased — and see §5, it is inflated at high frequency for a different
reason.) `rseek_B`'s deep tiling folds P > 0.25 s into ~1000 bins; we fold into
120. At 0.3% duty the pulse is under half a bin for us.

**The ladder is saturated there, which is the diagnostic signature.** Winning
rung vs duty, detected fundamentals: at duty <1% it is `H=60` (`k=1`) in 97–98%
of detections; 87% at 1–2%; 54% at 2–4%; by 8–16% the mass has moved to
`H=10–20`. Nothing deeper is available to win.

**And the population puts narrow duty exactly where deep folds are cheap.** The
TPA duty–period relation means 100% of the duty <0.5% injections are at
**f0 < 1.7 Hz** and 99.2% of duty <1% are below 12.5 Hz — nowhere near the
Nyquist ceiling of 8333 Hz / f0 harmonics.

## 4. Would `--nharms 120` close it? Yes — with `--maxdecim 12`, and it is a completeness fix, not an efficiency one

Analytic model, validated below: the recovered statistic of a band-limited
`M = 2H`-bin fold under the shipped boxcar bank is

```
snr1 = max_{w, phase} S_w / sqrt(var(S_w)),
S_w  = sum_h 2 Re(A_h W_h),   W_h = Dirichlet kernel of the w-bin window,
var(S_w) = sum_{h<H} 2|W_h|^2 + 0.5|W_H|^2     (the 1b8fed9 normalisation),
injected snr = sqrt(2 sum_{all h} |A_h|^2),
```

maximised over the ladder `H_k = nharms/k` and averaged over sub-bin pulse
phase. Two facts fall out immediately: at narrow duty the ladder efficiency and
the *ideal matched filter truncated at H* agree to 0.001 (0.591 vs 0.592 at 0.2%
duty), so **the boxcar bank contributes essentially nothing to the narrow-duty
loss — it is pure harmonic truncation**; and the model reproduces the measured
recovery to 1–6% in every duty bin (single global factor κ = 1.03, consistent
with the `hidr = 0.5` grid, the `m = 16` truncation and the analytic-σ bias).

Recovered-S/N ratio vs the current default, von Mises core:

| duty | H=60/k6 | H=120/k6 | **H=120/k12** | H=240/k24 |
|---|---|---|---|---|
| 0.2% | 0.591 | 0.795 | **0.795** | 0.941 |
| 0.3% | 0.709 | 0.901 | **0.901** | 0.994 |
| 0.5% | 0.859 | 0.979 | **0.979** | 0.979 |
| 0.7% | 0.929 | 0.991 | **0.991** | 0.991 |
| 1% | 0.979 | 0.979 | **0.979** | 0.994 |
| 5% | 0.994 | 0.970 | **0.994** | 0.994 |
| 8% | 0.993 | 0.947 | **0.993** | 0.993 |
| 12% | 0.958 | 0.936 | **0.958** | 0.958 |

**`--nharms 120` ALONE LOSES 2–5% at 5–12% duty.** With `maxdecim 6` the
shallowest rung becomes `H=20` (40 bins) instead of `H=10` (20 bins), and broad
pulses want the shallow rungs — this is the `(k, W)` duty-equivalence of
`ladder_boxcar_widths` seen from the other end. `--nharms 120 --maxdecim 12`
dominates the current default at every duty; `240/24` adds more below 0.3%.

Coverage and trial count are conserved by construction: `nharms x maxdecim`
fixes the top frequency, and fundamental trials `= (hifreq − lofreq)·T·nharms/hidr`,
so `120/12/hifreq 62.5` searches the same 0.1–750 Hz with the same number of
fundamentals — at ~2x the per-trial cost and 2x the (trial, rung) pairs.

**Predicted detection fraction, threshold held at 7.05** (the model, applied to
the drawn population):

| duty | N | measured | H60/k6 | H120/k12 | H240/k24 |
|---|---|---|---|---|---|
| <0.5% | 757 | 43.6% | 51.3% | **69.5%** | 73.1% |
| 0.5–1% | 4366 | 62.1% | 68.6% | 72.4% | 73.5% |
| 1–2% | 11228 | 69.1% | 72.1% | 73.0% | 73.1% |
| 2–4% | 16074 | 71.4% | 72.6% | 73.1% | 73.1% |
| 4–8% | 15944 | 73.1% | 73.7% | 73.7% | 73.8% |
| 8–16% | 15064 | 72.4% | 70.5% | 70.6% | 70.7% |
| >16% | 18581 | 71.6% | 68.0% | 68.0% | 68.1% |
| **all** | 82014 | 70.9% | 70.9% | **71.5%** | 71.7% |

**But the trials penalty eats it.** A blanket `120/12` is 2–4x the effective
trial count, i.e. `Δthr ≈ ln(N)/thr = +0.1…0.2`, and the overall figure goes
71.5% → **69.9% (+0.1)** → **68.2% (+0.2)**. *A blanket deep search is a net
loss.* Restricting the deep pass to low frequency, with a **per-region**
threshold (the `--normalize` item in §3 of `Summary_and_Future_Work.md` — the
deep tier must pay for its own trials, not tax the rest of the band):

| deep tier | extra compute | Δthr in tier | duty<0.5% | duty<1% | all |
|---|---|---|---|---|---|
| *baseline H60/k6* | — | — | 51.3% | 66.0% | **70.9%** |
| H120/k12 below 5 Hz | ~16% | +0.20 | 66.3% | 68.9% | 70.4% |
| H240/k24 below 1.7 Hz | ~20% | +0.39 | 66.5% | 67.5% | 70.2% |
| H240/k24 below 3 Hz | ~37% | +0.39 | 66.9% | 67.4% | 69.8% |
| H240/k24 below 12.5 Hz | ~159% | +0.39 | 66.9% | 67.2% | 68.7% |

**`H120/k12 below 5 Hz` is the pick**: ~16% more compute, duty<0.5% goes
51.3% → 66.3% (matching `rseek_B`'s measured 64.4%), and the overall figure
moves −0.5 points. So the honest framing for the paper is **completeness over a
real astrophysical population — narrow-duty slow pulsars are exactly what an FFA
is built for and we currently under-serve them — not "the headline number goes
up."** It does not.

**Caveat on the model's absolute level.** After the single global calibration it
reads 51.3% at duty <0.5% against a measured 43.6%, so it is optimistic by ~8
points in that one corner; the H120 prediction there is probably nearer 62% than
69.5%. The *ratios* between configurations are what it is good for. Run 2 should
measure this arm rather than trust it.

## 5. `prepfold_snr1` is inflated for MSPs — the `DOF_corr` problem (Scott, 2026-08-28)

PRESTO's `fold()` drizzles each finite-duration sample across every profile bin
it covers, which **correlates the profile bins**; `DOF_corr(dt_per_bin)` in
`src/fold.c` is the semi-analytic correction, `dt_per_bin = P/nbins/dt`. Our
`snr1()` on the `.bestprof` divides by `σ√(w(1−δ))` as if the bins were
independent. They are not, and the study can see it. At *fixed* duty (5–20%, so
profile-shape loss is flat):

| f0 | nbins | dt/bin | DOF_corr | prepfold_snr1/snr | coherent/snr |
|---|---|---|---|---|---|
| 0.1–1 Hz | 128 | 221 | 0.960 | 0.903 | 0.961 |
| 1–5 | 128 | 46.2 | 0.959 | 0.904 | 0.981 |
| 5–20 | 128 | 11.5 | 0.954 | 0.911 | 0.984 |
| 20–100 | 128 | 3.98 | 0.919 | 0.923 | 0.985 |
| 100–200 | 64 | 1.38 | 0.752 | 0.965 | 0.990 |
| 200–300 | 64 | 1.04 | 0.667 | **1.009** | 0.993 |
| 300–400 | 49 | 1.00 | 0.654 | **1.004** | 0.994 |
| 400–1000 | 36 | 1.00 | 0.654 | 0.968 | 0.988 |

prepfold's column *rises 11% with frequency and crosses 1.0*, tracking
`DOF_corr`; ours is flat at 0.96–0.99. Since prepfold is the ceiling column, an
inflated ceiling in the MSP band is the worst possible place for it.

**This is structurally the same bug as 1b8fed9** — a boxcar statistic normalised
as if the profile bins were independent. Ours are correlated because harmonics
past `H` are zero; prepfold's because samples are drizzled.

**The correction is width-dependent, and `sqrt(DOF_corr)` is only its wide-`w`
limit.** Simulating the drizzle directly (white samples spread over the bins
each covers; fold; measure `σ_bin·√(w(1−δ)) / sd(S_w)`, the factor `snr1` should
be multiplied by):

| nbins, dt/bin | w=1 | w=2 | w=4 | w=9 | w=19 | √DOF_corr |
|---|---|---|---|---|---|---|
| 128, 271 | 0.996 | 0.995 | 0.994 | 0.995 | 0.991 | 0.980 |
| 128, 12.9 | 0.996 | 0.990 | 0.986 | 0.983 | 0.987 | 0.977 |
| 128, 4.09 | 0.996 | 0.973 | 0.961 | 0.953 | 0.958 | 0.959 |
| 64, 1.39 | 0.992 | 0.921 | 0.894 | 0.881 | 0.886 | 0.868 |
| 64, 1.04 | 0.992 | 0.891 | 0.851 | 0.829 | 0.815 | 0.817 |

A one-bin boxcar cannot feel bin-to-bin correlation, so the factor is ~1.00 at
`w = 1` and falls to ~`√DOF_corr` for wide boxcars. Applying it flattens the
trend (200–300 Hz: 1.009 → ~0.84, against 0.89 at low frequency; the residual is
the shrinking `nbins` prepfold picks above 300 Hz). **Do not apply a flat
`√DOF_corr`** — it over-corrects narrow pulses by ~20% at `dt_per_bin ≈ 1`.

The fix belongs next to `snr1()` in `mc_simulate.py` as a
`drizzle_boxcar_corr(nbins, dt_per_bin)` table, keyed the way `_boxcar_shape!`
is. Simulate it once per `(nbins, dt_per_bin)` cell rather than deriving it —
`DOF_corr` itself is a Monte-Carlo fit, so there is no closed form to match.

## 6. What the analysis methodology got right, and what it did not

**Right, and worth keeping:**

* **False alarms counted on every realisation, not only the empty ones.**
  Verified: thresholds derived from injection-free realisations only agree with
  the all-realisation ones to ≤0.10 for every code at FAP 1e-2, and the fixed-cut
  rates agree to a few percent. The "candidate claimed by an injection at some
  `n/m`" deflation is negligible, so the larger sample is free.
* **Statistic values recorded, not booleans.** Every table above that is not the
  quick-look's was built from the same `.jsonl` with no re-running.
* **`--fap` matching.** A nominal cut is meaningless here: the four codes sit at
  8.00 / 7.95 / 8.05 / **7.05** for the same false-alarm rate.
* **Counting harmonic detections as detections.** 98.9% of ours are labelled `1`
  against 93.6% for `accelsearch`; scoring the family as misses would have been a
  1–5% artefact in the wrong direction.

**Wrong or missing:**

* **The subset trap is not fully fixed by the `*` footnote** (§1). Cross-code
  cells must be restricted to realisations every compared code ran.
* **`rseek_B`'s 200-entry false-alarm tail is saturated in 100% of
  realisations**, so its FA *rate* curve is a ceiling below 6.5. Harmless at the
  thresholds actually used (its rate at 6.5 is 48.4, well under the cap) but the
  script should say so rather than leave it to be noticed.
* **Errors are computed per injection.** Six injections share one noise
  realisation, so they are not independent; bootstrap by *realisation*.
* **`accelsearch` is run on the raw `.fft` while we get `_red.fft`** — an
  asymmetry a referee will find. Run it on both.
* **1-D marginals confound duty and frequency.** In the TPA population the two
  are strongly correlated (median duty 1.12% at 0.1–0.3 Hz, 17.5% at
  100–400 Hz), so "we lose at narrow duty" is also "we lose below 2 Hz".

## 7. Wanted in `mc_analyze.py`

1. Restrict every cross-code cell to the realisations all compared codes ran;
   emit the paired table, not two marginals.
2. Bootstrap by realisation, not by injection.
3. **McNemar / discordant-pair counts** per cell. `1054 vs 181` says far more
   than `71.6% vs 64.3%`, and paired noise is what the design bought.
4. **ROC (detection fraction vs FAP), not one matched threshold.** That the
   ordering is invariant from FAP 1 to 1e-3 is itself a result (§1).
5. Flag top-N saturation in the false-alarm tails.
6. **Decompose the advantage into threshold + paired Δstatistic**, per bin (§2).
7. **Report each statistic's null scatter** at fixed injected S/N (§2b). This is
   where the analytic σ shows up and it appears nowhere in the current output.
8. **2-D maps (duty × f0) and (duty × S/N)**, not just 1-D marginals.
9. Report both TPA-weighted and flat-in-log-duty numbers, so the headline is not
   an artefact of where the population sits.
10. **Sensitivity and cost on one axis** — detection fraction per CPU-second, or
    "injected S/N at 50% detection at fixed FAP and fixed cost". §3.2 asks for
    this and it is not there yet.
11. Overlay the §4 analytic efficiency model on the measured curves; it matched
    to a few percent, so a departure is a bug signal.
12. Duty recovery as a bias + scatter table, against the realised `ducy_got`.
13. Apply the §5 drizzle correction to `prepfold_snr1`, and put prepfold into the
    FAP-matched table with a measured null instead of a nominal 6.0.

## 8. Wanted in run 2

**New arms (cheap, high value)**

* **A `--nharms 120 --maxdecim 12 --hifreq 62.5` coherent arm**, and/or a
  low-frequency-only deep tier. It is one more `coherent_search.jl` invocation
  and it is the single most valuable new column — §4 is a prediction until it is
  measured.
* **Run `accelsearch` on `_red.fft` as well as `.fft`.**
* Consider `--deep-every 5`: `rseek_B` is 121 s and already losing.

**Recording**

* **Store prepfold's profile itself** (or at minimum `nbins`, `dt_per_bin` and
  the winning width) so §5's correction can be applied post hoc without
  re-folding. And compute its σ analytically (`√(N/nbins)` on normalised noise)
  rather than from a 128-bin MAD — that MAD alone costs 1.9 sd of statistic
  scatter and makes the reference look worse than the reference is.
* **Fold the injection-free realisations with prepfold at random periods** drawn
  from the same population. That gives a measured null for `snr1` and
  `chi2_sigma`, and lets prepfold join the FAP-matched table as a proper ceiling
  rather than a nominal-threshold aside.
* **Record `rseek`'s fold depth `b` per candidate.** We record `nharm` (hence
  `k`); without `b` we cannot tell whether an `rseek` miss at narrow duty is the
  downsampling sawtooth or the `bins_min` width bank.
* **Record each code's trial count** so cost can be normalised per trial.
* Record the §4 model efficiency per injection (~1 ms from the drawn
  `(ducy, w10_w50)`) so recovery can be normalised out.

**Sampling**

* **Stratify duty and record importance weights.** 118 injections below 0.5%
  duty is not enough to settle the one place we lose; target ≥1000 there and
  reweight to the TPA population at analysis time.
* **Continuous injected S/N** over ~5.5–11.5 rather than six integers, so the
  efficiency curve can be fitted. The current grid brackets well (15% at 6,
  99.6% at 11) but wastes resolution.

**New axes, one run each**

* **Red noise.** Everything here is pure white noise — the regime where the
  analytic σ is exactly right and where riptide's per-profile σ̂ has no
  compensating advantage. Real data is the reason `--sigma measured` exists.
  This is the most valuable *new* axis and the one a referee is most likely to
  ask about.
* **A second observation length.** All of this is `T = 1006 s`. The threshold
  advantage of §2a should grow with `T`; worth one confirming run.
