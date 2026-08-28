#!/usr/bin/env python3
"""
mc_profiles.py -- the injected pulse population for the sensitivity Monte Carlo.

Two things live here, kept apart from the pipeline driver because they are the
part with actual astronomy in them:

  * the (period, duty cycle, profile shape) SAMPLER, drawn from the MeerKAT
    Thousand-Pulsar-Array width measurements (Posselt et al. 2021, MNRAS 508,
    4249) rather than from a guessed distribution;
  * the PROFILE MODEL, a von Mises core with a one-sided exponential scattering
    tail whose timescale is solved so the profile reproduces the drawn pulsar's
    measured W10/W50 -- i.e. the wings come from data too.

Why not just a von Mises, as riptide's `generate_signal` builds?  Because the TPA
sample says a single von Mises cannot represent about a third of the population.
Its W10/W50 is pinned at 1.823 (the Gaussian value); the measured ratio spans
1.18 to 10.6, quartiles 1.66-2.86, with 23% above 3.0.  Since harmonic content is
exactly what separates a 60-harmonic coherent sum from a 16-harmonic incoherent
one, getting the wings wrong would bias the headline comparison -- and it would
bias it in OUR favour, because broad wings carry less high-harmonic power.

**The injected S/N is defined against a ZERO-MEAN unit-L2 template**, not
riptide's convention.  `generate_signal` normalises the von Mises including its
DC component, but every search removes the baseline, so that convention makes a
duty-cycle-dependent fraction of the "injected" S/N unrecoverable in principle:
0.992x at 1% duty but 0.921x at 10%, 0.827x at 20% and 0.696x at 30%.  Duty cycle
is the main axis of this study, so that would have put a 30% artifact right along
it.  Here "injected S/N 9" means an ideal baseline-removing matched filter that
knows the pulse shape gets 9.
"""

from __future__ import annotations

import math
import numpy as np

# --- TPA width sample -------------------------------------------------------
# Fitted directly to the supplementary table_1.csv W50 column (Gflag==1, >1 deg),
# N=1176, P = 0.036-7.73 s.  Deriving this from the paper's published W10 fits
# instead is wrong by ~2x in the narrow tail: real W10/W50 is 2.10, not the
# Gaussian 1.823, and the residual scatter is 0.318 dex and RIGHT-SKEWED, not the
# 0.225 dex a symmetric fit to their 5%/10% quantile lines implies.
TPA_SLOPE = -0.3472           # log10(ducy) vs log10(P/s)
TPA_INTERCEPT = -1.7194       # => median FWHM duty 1.91% at P = 1 s
TPA_RESID_SD = 0.318          # dex; only used if no empirical residuals are given

# Quantiles of the measured log-residual, so the skew survives even without the
# CSV to hand.  (5,10,25,50,75,90,95) percentiles, dex.
TPA_RESID_Q = np.array([-0.480, -0.383, -0.220, -0.020, 0.190, 0.404, 0.560])
TPA_RESID_P = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

# Measured W10/W50 quantiles, same sample (N=824 with both widths good).
TPA_RATIO_Q = np.array([1.18, 1.26, 1.34, 1.66, 2.10, 2.86, 4.31, 10.64])
TPA_RATIO_P = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])

VONMISES_W10_W50 = math.sqrt(math.log(10.0) / math.log(2.0))   # 1.8226


def load_tpa(path):
    """Read the TPA supplementary table_1.csv -> (P, ducy50, ratio) arrays.

    Returns the *empirical* sample so the driver can bootstrap real residuals and
    real W10/W50 pairs.  Scattered pulsars (`Sflag`) are KEPT: the paper excludes
    them from its width fits, but they are real pulsars a survey has to find, and
    dropping them biases the population narrow -- which flatters a deep harmonic
    sum.  Their median duty is 3.71% against 2.15% for the rest.
    """
    import csv
    P, duty, ratio = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                p = float(row["P"]); w50 = float(row["W50"])
            except (TypeError, ValueError):
                continue
            if row.get("W50Gflag") != "1.0" or w50 <= 1.0:
                continue
            P.append(p); duty.append(w50 / 360.0)
            try:
                w10 = float(row["W10"])
                ratio.append(w10 / w50 if row.get("W10Gflag") == "1.0" and w10 > 1
                             else float("nan"))
            except (TypeError, ValueError):
                ratio.append(float("nan"))
    P = np.array(P); duty = np.array(duty); ratio = np.array(ratio)
    resid = np.log10(duty) - (TPA_SLOPE * np.log10(P) + TPA_INTERCEPT)
    return dict(P=P, duty=duty, ratio=ratio, resid=resid,
                ratio_ok=ratio[np.isfinite(ratio)])


class Population:
    """Draws (period, duty, W10/W50) for one injected pulsar.

    `tpa` is `load_tpa(...)` or None; without it the residual and ratio are drawn
    from the recorded quantiles instead, which keeps the sampler usable on a host
    that does not have the CSV.
    """

    def __init__(self, tpa=None, msp_frac=1.0 / 3.0,
                 slow_p=(0.02, 5.0), msp_p_mean=3.5e-3, msp_p_sd=1.0e-3,
                 msp_p_range=(1.5e-3, 10.0e-3), msp_ducy=(0.02, 0.30)):
        self.tpa = tpa
        self.msp_frac = msp_frac
        self.slow_p = slow_p
        self.msp_p_mean = msp_p_mean
        self.msp_p_sd = msp_p_sd
        self.msp_p_range = msp_p_range
        self.msp_ducy = msp_ducy

    def _resid(self, rng):
        if self.tpa is not None:
            return float(rng.choice(self.tpa["resid"]))
        return float(np.interp(rng.random(), TPA_RESID_P, TPA_RESID_Q))

    def _ratio(self, rng):
        if self.tpa is not None and len(self.tpa["ratio_ok"]):
            return float(rng.choice(self.tpa["ratio_ok"]))
        return float(np.interp(rng.random(), TPA_RATIO_P, TPA_RATIO_Q))

    def draw(self, rng, dt, min_fwhm_samples=3.0):
        """One pulsar.  Rejects until the pulse is resolved by the sampling.

        A FWHM under a few samples is not a meaningful injection: point-sampling
        a narrower pulse makes its sampled shape depend on the sub-sample phase,
        so no filter can match it and "injected S/N" stops meaning one thing.
        The rejection is recorded (`ducy_floor`) rather than silently applied.
        """
        for _ in range(200):
            msp = rng.random() < self.msp_frac
            if msp:
                P = float(rng.normal(self.msp_p_mean, self.msp_p_sd))
                if not (self.msp_p_range[0] <= P <= self.msp_p_range[1]):
                    continue
                # TPA has no MSPs; theirs are systematically broader, so this is
                # a stated prior rather than a measured distribution.
                ducy = float(rng.uniform(*self.msp_ducy))
                ratio = VONMISES_W10_W50      # no wing data for MSPs: keep it simple
            else:
                lp = rng.uniform(math.log10(self.slow_p[0]), math.log10(self.slow_p[1]))
                P = float(10.0 ** lp)
                ducy = float(10.0 ** (TPA_SLOPE * lp + TPA_INTERCEPT + self._resid(rng)))
                ratio = self._ratio(rng)
            ducy = min(ducy, 0.45)
            if ducy * P < min_fwhm_samples * dt:
                continue                       # unresolved: redraw
            return dict(period=P, ducy=ducy, w10_w50=ratio, msp=bool(msp))
        raise RuntimeError("population draw failed to converge")


# --- profile model ----------------------------------------------------------
_NPH = 1 << 13        # phase grid for shape solving and for sampling the pulse

# A one-sided exponential tail cannot make W10/W50 exceed ln(10)/ln(2) = 3.322,
# because far out the profile IS the exponential and both widths are set by it.
# The measured TPA ratio reaches 10.6 and 23% of it is above 3.0, so the upper
# population is not scattering at all -- it is multi-component profiles, where
# W10 catches a second component that W50 misses.  Above the scattering limit we
# therefore add a broad, weak PEDESTAL component (a second von Mises at the same
# phase, `_PED_WIDE` times wider) and solve its amplitude instead.  Dropping
# those pulsars would bias the injected population NARROW, which flatters a deep
# harmonic sum -- the same trap as excluding the scattered ones.
_SCATTER_RATIO_MAX = math.log(10.0) / math.log(2.0)
# Width factor and amplitude bracket chosen from a monotonicity scan: at
# `_PED_WIDE = 12` the ratio rises monotonically from 1.83 to 10.6 over
# amplitude 0..0.45 -- covering the whole measured range -- and then TURNS OVER
# as the pedestal starts setting the 50% level too.  Bisecting past the turnover
# silently returns the bracket end, which is what a first attempt over 0..2 did.
_PED_WIDE = 12.0
_PED_MAX = 0.45


def _vonmises(phase, ducy):
    kappa = math.log(2.0) / (2.0 * math.sin(math.pi * min(ducy, 0.49) / 2.0) ** 2)
    # exp(kappa*(cos-1)) is already peak-normalised and never overflows.
    return np.exp(kappa * (np.cos(2 * math.pi * phase) - 1.0))


def _scatter(shape, tau):
    """Circularly convolve with a one-sided exponential of timescale `tau` (in
    phase units).  `tau <= 0` returns the shape untouched."""
    if tau <= 0:
        return shape
    n = len(shape)
    t = np.arange(n) / n
    k = np.exp(-t / tau)
    k /= k.sum()
    return np.fft.irfft(np.fft.rfft(shape) * np.fft.rfft(k), n)


def _width_at(shape, frac):
    """Width of a peak-normalised profile at `frac` of its peak, in phase units.

    Measured the way the TPA pipeline defines W50/W10: the full extent between
    the outermost crossings of the level, so a scattering tail counts.
    """
    n = len(shape)
    s = shape - shape.min()
    s = s / s.max()
    # Rotate the peak to the centre so the crossings do not wrap.
    s = np.roll(s, n // 2 - int(np.argmax(s)))
    above = np.nonzero(s >= frac)[0]
    if len(above) < 2:
        return 1.0 / n
    return (above[-1] - above[0] + 1) / n


def make_profile(ducy, w10_w50, nph=_NPH):
    """Peak-normalised profile with FWHM = `ducy` and the requested W10/W50.

    Solves the scattering timescale for the ratio (it is monotone in tau), then
    rescales the core so the FWHM lands on target.  Falls back to a plain von
    Mises if the ratio is at or below what one already gives.

    Returns `(profile, info)`; `info` records what was actually achieved, because
    a ratio in the far tail (the sample reaches 10.6) is not always reachable at
    a wide duty cycle and the analysis must be able to see that.
    """
    target = float(w10_w50)
    if not np.isfinite(target) or target <= VONMISES_W10_W50 * 1.02:
        p = _vonmises(np.arange(nph) / nph, ducy)
        return p, dict(tau=0.0, ped=0.0, ratio=VONMISES_W10_W50, ducy=_width_at(p, 0.5),
                       exact=abs(target - VONMISES_W10_W50) < 0.05)

    def build(core, tau):
        return _scatter(_vonmises(np.arange(nph) / nph, core), tau)

    def ratio_of(p):
        return _width_at(p, 0.1) / max(_width_at(p, 0.5), 1e-9)

    # The two solves are COUPLED -- scattering widens W50, so shrinking the core
    # to recover the FWHM also shrinks the tail and moves the ratio back.  Solve
    # them alternately rather than once each; three passes is well converged.
    # `tau` is carried in units of the core FWHM, which makes the bracket scale
    # free.  The upper bound has to be generous: at 2% duty a ratio of 5 needs
    # tau ~ 30x the core, and clipping it silently caps the achieved ratio near
    # 2.8 (which is what a bound of 4 did).
    # Below the scattering limit, solve the tail; above it, pin the tail just
    # under the limit and solve the pedestal amplitude for the rest.
    scattering = target < 0.98 * _SCATTER_RATIO_MAX

    def full(core, tau, ped):
        base = _vonmises(np.arange(nph) / nph, core)
        if ped > 0:
            base = base + ped * _vonmises(np.arange(nph) / nph,
                                          min(_PED_WIDE * core, 0.49))
            base /= base.max()
        return _scatter(base, tau)

    def solve1(fn, target_val, hi0):
        lo, hi = 0.0, hi0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if fn(mid) < target_val:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-5 * max(1.0, hi):
                break
        return 0.5 * (lo + hi)

    # The solves are COUPLED -- scattering and a pedestal both widen W50, so
    # recovering the FWHM shrinks the core and moves the ratio back.  Alternate
    # rather than solving each once.
    # One mechanism or the other, never both: combining them is not monotone in
    # either parameter, and the solve has to stay monotone to be bisectable.
    core, tau_rel, ped = ducy, 0.0, 0.0
    for _ in range(3):
        if scattering:
            tau_rel = solve1(lambda t: ratio_of(full(core, t * core, 0.0)), target, 60.0)
        else:
            ped = solve1(lambda a: ratio_of(full(core, 0.0, a)), target, _PED_MAX)
        for _ in range(40):
            got = _width_at(full(core, tau_rel * core, ped), 0.5)
            if got <= 0 or abs(got - ducy) < 1e-4 * ducy:
                break
            core *= ducy / got
            if core < 1e-5:
                break
    p = full(core, tau_rel * core, ped)
    got_r = ratio_of(p)
    # `exact` is False when the requested ratio was not reachable at this duty
    # cycle -- a very wide pulse cannot also have a very long tail without the
    # tail wrapping the period.  The analysis needs to see that, not have it
    # quietly clipped.
    return p, dict(tau=tau_rel * core, ped=ped, ratio=got_r, ducy=_width_at(p, 0.5),
                   exact=bool(abs(got_r - target) < 0.05 * target))


def inject(x, dt, period, phase0, profile, snr):
    """Add one pulsar to the time series `x` in place; return its true amplitude.

    The template is sampled, then made ZERO MEAN and scaled to unit L2, so `snr`
    is exactly what an ideal baseline-removing matched filter would recover
    against unit-variance noise.  Doing the normalisation on the SAMPLED template
    (rather than analytically) also means a narrow pulse's injected S/N is exact
    for the realisation it actually got, not for an idealisation of it.
    """
    n = len(x)
    nph = len(profile)
    # Wrap ONCE, by padding, instead of two `% nph` passes over 16M elements:
    # after `% 1.0` the index is already in [0, nph), so only `i0 + 1` can run
    # off the end, and a duplicated first sample covers it.
    prof = np.empty(nph + 1, dtype=np.float64)
    prof[:nph] = profile
    prof[nph] = profile[0]

    # Blocked, and in two passes rather than six.  The naive form materialises
    # five 134 MB temporaries per injection and cost 4.2 s each -- 25 s of a
    # 111 s realisation, for no science.  Accumulating the sum and sum of
    # squares in the first pass lets the zero-mean unit-L2 scaling be applied in
    # the second without ever forming `s - mean` as a separate array.
    step = 1 << 20
    s = np.empty(n, dtype=np.float64)
    rate = dt / period
    tot = 0.0
    tot2 = 0.0
    for a in range(0, n, step):
        b = min(a + step, n)
        ph = np.arange(a, b, dtype=np.float64)
        ph *= rate
        ph += phase0
        ph %= 1.0
        ph *= nph
        i0 = ph.astype(np.int64)          # non-negative, so truncation IS floor
        f = ph - i0
        blk = prof[i0]
        blk += f * (prof[i0 + 1] - blk)   # linear interp, one temporary
        s[a:b] = blk
        tot += float(blk.sum())
        tot2 += float(np.dot(blk, blk))

    mean = tot / n
    var = tot2 - n * mean * mean          # == sum((s-mean)^2)
    if var <= 0:
        return 0.0
    scale = snr / math.sqrt(var)
    off = mean * scale
    for a in range(0, n, step):
        b = min(a + step, n)
        blk = s[a:b]
        blk *= scale
        blk -= off
        x[a:b] += blk
    return float(scale)
