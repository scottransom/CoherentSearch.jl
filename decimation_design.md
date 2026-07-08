# Cheap Multi-Frequency Search by Harmonic Decimation

Design notes for extending the CoherentSearch.jl hot loop to search integer
*multiples* of every trial fundamental almost for free, by re-using the
harmonic amplitudes it already interpolated. Companion to
`coherent_search_design.md` (which covers the base search) and item
"Cheap multi-frequency search by harmonic decimation" in
`Summary_and_Future_Work.md`.

 - Base search: `src/search.jl`, function `search` (the chunk-parallel hot loop).
 - Python reference: `/home/sransom/git/coherent_search`.

---

## 1. The idea

For a base trial fundamental Fourier frequency `rf`, the search already
interpolates the complex amplitude of every harmonic `h = 1 … nharms` at the
frequencies `rf·1, rf·2, …, rf·nharms`, stacks them (row `h+1`, DC in row 1) in
`ftprofs`, and inverse-real-FFTs the stack into a `2·nharms`-bin pulse profile.
The interpolation is the expensive part; the `irfft` and the metric are cheap.

To search the multiple `k·rf` we do **not** need any new interpolation. The
harmonics of a signal at fundamental `k·rf` fall at `k·rf, 2k·rf, 3k·rf, …` —
which are exactly the amplitudes we *already computed* at base-harmonic numbers
`k, 2k, 3k, …`. So we take **every k-th** row of `ftprofs`, pack them into a
shorter stack (they become harmonics `1, 2, …, Hₖ` of the multiple, where
`Hₖ = ⌊nharms/k⌋`), and inverse-real-FFT that into a `2·Hₖ`-bin profile. One
strided copy plus a short `irfft` per `k` — the interpolation is amortised
across every decimation.

This targets **faster pulsars** (higher fundamental frequency), which also tend
to have wider duty cycles and therefore need *fewer* harmonics — exactly what a
higher `k` provides (`Hₖ` shrinks with `k`).

---

## 2. Why it is nearly free (and correctly sampled)

Three properties make decimation cheap *and* safe, and they answer the caveats
raised in the future-work note ("the step in fundamental frequency … and how far
each chunk reads into the long FFT is tuned for the base `Nbins`"):

**(a) Top-harmonic sampling is preserved automatically.** The base grid steps
the fundamental by `deltar = hidr/nharms` bins, chosen so the *highest* harmonic
(`nharms`) advances by `hidr` (default 0.5) bins per trial — the anti-aliasing
constraint. A decimated fundamental `k·rf` advances by `k·deltar`, but its
*own* highest harmonic (number `Hₖ`) advances by `Hₖ·k·deltar ≤ nharms·deltar =
hidr`. So every decimation is sampled at **≤ hidr bins at its top harmonic** —
never coarser than the base search, so no signal is missed. `deltar`,
`numbetween`, and the per-harmonic grids need **no per-`k` retuning**.

  (When `k ∤ nharms`, `Hₖ·k < nharms`, so that pass is sampled a touch *finer*
  than `hidr` and the few base amplitudes between `Hₖ·k` and `nharms` simply go
  unused for that `k` — a small waste of amplitudes we already have in hand, and
  the price of letting `k` be any integer rather than only divisors of
  `nharms`.)

**(b) The input-FFT read depth is unchanged.** Every decimated top harmonic sits
at `Hₖ·(k·rf) = (Hₖ·k)·rf ≤ nharms·rf` — a Fourier frequency the base search
*already* reads (and range-checks against Nyquist in `fill_harmonic_row!`). So
reading out to `nharms·rf` covers **all** decimations. To search fundamentals up
to `kmax·hifreq` we pay input reads only out to `nharms·hifreq`. Harmonics that
run past Nyquist are already left at zero by the base range check, and a
decimated stack inherits those zeros correctly.

**(c) Cross-`k` duplicates collapse for free.** A real pulsar at Fourier
frequency `r = f·T` is reported at `r_dec = k·rf ≈ f·T` in *every* pass `k` for
which `f/k` lies in the searched band: pass `k` picks the trial `rf` nearest
`f·T/k`, so `k·rf` lands within `k·deltar ≲ 0.05` bins of `f·T` (for `k ≤ 6`,
`nharms = 60`). All passes therefore report the same `r` to far better than the
existing `dr_tol = 1.0` bin, and `remove_duplicates` already keeps the
strongest — normally the `k = 1` fold with the most harmonics. **No new
dedup logic is required**; a pulsar detected across several decimations yields
one candidate.

The genuine cost of decimation is simply that it wants a **larger, more
composite `nharms`** (so many `k` give clean integer `Hₖ`): the base
interpolation then generates more harmonics per trial — paid once — while each
extra decimation is only a strided copy + a short `irfft` + the metric scan.

---

## 3. Bookkeeping (`nharms = 60`, `hidr = 0.5`, `kmax = 6`)

`60` is divisible by `1…6`, so the `⌊⌋` never bites in the default range:

| k | base rows used (harmonic `h`) | Hₖ = ⌊nharms/k⌋ | Nbins = 2Hₖ | reported freq | fundamentals covered |
|---|---|---|---|---|---|
| 1 | 1, 2, …, 60  | 60 | 120 | rf / T   | [lo, hi]   |
| 2 | 2, 4, …, 60  | 30 | 60  | 2·rf / T | [2lo, 2hi] |
| 3 | 3, 6, …, 60  | 20 | 40  | 3·rf / T | [3lo, 3hi] |
| 4 | 4, 8, …, 60  | 15 | 30  | 4·rf / T | [4lo, 4hi] |
| 5 | 5, 10, …, 60 | 12 | 24  | 5·rf / T | [5lo, 5hi] |
| 6 | 6, 12, …, 60 | 10 | 20  | 6·rf / T | [6lo, 6hi] |

Per-`k` quantities that change: the harmonic count `Hₖ`, the profile length
`2Hₖ` (hence the `irfft` plan), the noise term
`ngoodbins = min(N/2 / (k·rmean), Hₖ)`, the reported Fourier frequency
`r_dec = k·rf` (and its Hz `r_dec/T`, period `T/r_dec`), and the harmonic count
attached to the candidate.

---

## 4. Implementation sketch

Only the optimised `search` path changes; the reference `block_metrics` /
`reference_profiles` stay `k = 1` and remain the oracle-pinned audit path.

- **`SearchParams`** gains `decimations::Vector{Int}` (default `[1]` — current
  behaviour, zero overhead). A helper `decimation_set(nharms, maxdecim)` returns
  `[k for k in 1:maxdecim if ⌊nharms/k⌋ ≥ 2]`.

- **`Candidate`** gains `nharm::Int`: the number of harmonics summed for the
  detection (`Hₖ`). This *is* the decimation label (`k = nharms ÷ nharm`), so
  the CLI can report it directly. Period is `1/freq`.

- **Per-`k` buffers + plans (`DecimBuf`).** For each `k ≥ 2`: a compact
  `dftprofs` `(Hₖ+1, Nprof)`, a real `dprofs` `(2Hₖ, Nprof)`, a `medbuf`, and a
  batched `plan_brfft(dftprofs, 2Hₖ, 1)`. Built once per `Workspace`
  (single-threaded, like the existing base plan), private per task. When
  `decimations == [1]` the list is empty and the base path is byte-for-byte
  unchanged (preserving the `align=false` equivalence test).

- **Hot loop.** After the existing base (`k = 1`) profile + metric pass, for each
  `DecimBuf`: gather the strided rows (`dftprofs[j+1, :] .= ftprofs[j·k+1, :]`
  for `j = 1…Hₖ`; row 1 stays DC = 0), one `mul!(dprofs, brfftplan, dftprofs)`,
  then the same `_profile_snr` scan with `invrms` from the `k`-specific
  `ngoodbins` and `scale = 1/(2Hₖ)`. Emit `Candidate(k·rf/T, metric, k·rf, Hₖ)`
  for trials above threshold, skipping any with `r_dec ≥ N/2` (fundamental past
  Nyquist). The strided copy is `O((Hₖ+1)·Nprof)`, negligible beside the
  `nharms` interpolations.

- **CLI.** Add `--maxdecim` (default 1 = off). When `> 1`, set
  `decimations = decimation_set(nharms, maxdecim)` and default `nharms` to `60`
  (composite) unless the user set it explicitly. Report frequency **and** period
  **and** harmonic count per candidate. `remove_duplicates` stays on by default
  so cross-`k` duplicates collapse.

---

## 5. Validation

- **Native-fold equivalence (the strong test, no new Python needed).**
  Decimation pass `k` must reproduce, to machine precision, a *native* `Hₖ`-
  harmonic fold at the multiplied frequencies — i.e. the strided-gather +
  short `irfft` must equal `reference_profiles(ft, k·rfund,
  SearchParams(nharms=Hₖ, …))`. Since `reference_profiles` is already pinned to
  the Python oracle at ~8e-16, this transitively pins decimation. One asserted
  subtlety: the decimated fold drops the imaginary part of its own top harmonic
  (its Nyquist bin), which the deeper base fold does not — so the equivalence is
  against the *native* `Hₖ`-harmonic fold, which drops it identically.

- **Detection via decimation, same data.** The bundled `harmonics_hi.fft` pulsar
  at `10.0123 Hz` is recovered by a *base* band chosen so it only triggers via a
  higher `k`: `--lofreq`/`--hifreq` around `10.0123/k` (e.g. `[4, 6]` → `k = 2`,
  `[3.2, 3.5]` → `k = 3`). Assert the reported candidate is `≈ 10.0123 Hz` with
  `nharm = ⌊nharms/k⌋`, and that with decimation off the same band finds nothing.

- **Thread-count / chunk-size invariance** of the candidate list, as in the
  existing suite.

---

## 6. Open items / future tuning

- **Threshold comparability across `Hₖ`.** The metric's numeric scale depends on
  the harmonic count (via `ngoodbins`) just as it already depends on
  `--metric`/`--pexp` (see the calibration item in `Summary_and_Future_Work.md`).
  Keeping the max-metric member across `k` is reasonable, but a principled
  per-`Hₖ` threshold (or a trials-corrected significance) is the same open
  calibration problem, now with `k` as an extra axis.

- **Harmonically-related (not just identical) duplicates.** Decimation makes the
  base `f`, `2f`, `3f` structure of a real signal even more visible across
  passes; a proper harmonic-summing de-duplication (distinct from the
  near-identical-`r` collapse) is still future work.

- **`kmax` vs. Nyquist and `nharms` choice.** `kmax` is bounded by
  `⌊nharms/k⌋ ≥ 2` and by the Nyquist cap on `k·rf`; the sweet spot for `nharms`
  (60 vs. 120 vs. …) trades base interpolation cost against decimation depth and
  should be swept alongside the `fftlen`/`numbetween` throughput tuning.
