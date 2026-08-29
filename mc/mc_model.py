#!/usr/bin/env python3
"""
mc_model.py -- the two analytic models the Monte Carlo needs.

Kept apart from both the driver and the analysis because both of them use these,
and because each one exists to answer a question that came out of run 1
(`docs/monte_carlo.md` sections 4 and 5).

  1. `ladder_efficiency` -- what fraction of an injected signal's S/N a
     BAND-LIMITED coherent fold can recover.  This is section 4's model: it predicts
     the measured recovery to a few percent, so a departure from it in the data
     is a bug signal rather than astronomy, and it is what turns "we lose below
     0.5% duty" into "we lose below 0.5% duty BECAUSE the harmonic sum truncates
     there, and `--nharms 120 --maxdecim 12` fixes it".

  2. `drizzle_boxcar_corr` -- the factor `snr1` on a *prepfold* profile must be
     multiplied by, because PRESTO's `fold()` drizzles each finite-duration sample
     across every profile bin it touches and so CORRELATES the bins.  Without it
     prepfold's column rises 11% with spin frequency and crosses 1.0 -- an
     inflated ceiling, in the MSP band, which is the worst possible place for one.

Both are deliberately cheap: `ladder_efficiency` is ~1 ms, so the driver can
record it per injection, and `drizzle_boxcar_corr` is exact linear algebra on the
fold weights (no Monte Carlo), memoised per `(nbins, samples-per-bin)` cell.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# The ladder: mirrors `decimation_set` / `boxcar_widths` / `ladder_boxcar_widths`
# in src/search.jl.  Kept as a transcription rather than derived, so that a
# change on the Julia side shows up here as a disagreement instead of being
# silently modelled wrong; `test_mc.py` pins the three against the Julia.
# ---------------------------------------------------------------------------
_LADDER_WMAX = 6


def decimation_set(nharms, maxdecim):
    return [k for k in range(1, max(1, maxdecim) + 1) if nharms // k >= 2]


def boxcar_widths(nbins, fsp=1.5, maxfrac=0.3):
    wmax = min(max(1, int(maxfrac * nbins)), int(nbins) - 1)
    out, w = [], 1
    while w <= wmax:
        out.append(w)
        w = max(int(fsp * w), w + 1)
    return out


def ladder_boxcar_widths(nbins, k, ks, fsp=1.5, maxfrac=0.3):
    """The pruned bank fold `k` actually scans (see `ladder_boxcar_widths` in
    src/search.jl for why most of the (k, W) grid is redundant)."""
    ws = boxcar_widths(nbins, fsp, maxfrac)
    ks = sorted(ks)
    if len(ks) <= 1:
        return ws
    if not all(ks[i + 1] <= (_LADDER_WMAX // 2) * ks[i] for i in range(len(ks) - 1)):
        return ws
    kmin, kmax = ks[0], ks[-1]
    pruned = [w for w in ws
              if (w >= 2 or k == kmin) and (w <= _LADDER_WMAX or k == kmax)]
    return pruned or ws


# ---------------------------------------------------------------------------
# Section 4: recovered S/N of a band-limited fold
# ---------------------------------------------------------------------------
def profile_harmonics(prof, hmax):
    """Complex Fourier amplitudes `A_h`, h = 1..hmax, of a profile, normalised so
    that `2 * sum_h |A_h|^2 == 1` over ALL harmonics.

    That normalisation is the definition of the injected S/N used throughout the
    study (`mc_profiles.inject` scales the sampled template to unit L2 after
    removing its mean), so the number this module returns is directly a FRACTION
    of injected S/N -- no calibration constant, and none is fitted.
    """
    p = np.asarray(prof, dtype=float)
    n = len(p)
    A = np.fft.rfft(p)[1:] / n           # h = 1 .. n//2
    tot = 2.0 * float(np.sum(np.abs(A) ** 2))
    if tot <= 0:
        return np.zeros(hmax, dtype=complex)
    A = A / math.sqrt(tot)
    out = np.zeros(hmax, dtype=complex)
    m = min(hmax, len(A))
    out[:m] = A[:m]
    return out


def _dirichlet(h, w, M):
    """|D_h(w)| e^{i phi}: the response of a w-bin boxcar on an M-bin profile at
    profile harmonic h.  `h` and `w` broadcast."""
    num = np.sin(np.pi * h * w / M)
    den = np.sin(np.pi * h / M)
    mag = np.where(np.abs(den) < 1e-12, np.broadcast_to(np.asarray(w, dtype=float), num.shape),
                   num / np.where(np.abs(den) < 1e-12, 1.0, den))
    # The boxcar starts at bin 0, so it carries a linear phase; keeping it means
    # the phase maximisation below is over the boxcar's LEFT edge, which is what
    # the search scans.
    return mag * np.exp(1j * np.pi * h * (w - 1) / M)


def ladder_efficiency(A, nharms=60, maxdecim=6, hmax_data=None, nsub=4,
                      fsp=1.5, maxfrac=0.3):
    """Fraction of an injected signal's S/N the ladder recovers, and which rung.

    `A` is `profile_harmonics(...)` (long enough for `nharms`).  `hmax_data` is
    the highest harmonic present in the DATA -- `floor(f_nyquist / f0)` -- past
    which `fill_harmonic_row_direct!` gives up and leaves the row zero; pass it
    or the model will happily fold harmonics the sampling never recorded.

    Returns `(efficiency, k, width, nbins)` for the rung that wins on average.

    The statistic modelled is exactly the shipped one (`_boxcar_shape!`):

        S_w      = sum_{h filled} 2 Re(A_h D_h(w) e^{2 pi i h s / M})
        var(S_w) = sum_{h filled, h < M/2} 2 |D_h|^2  +  (1/2) |D_{M/2}|^2

    maximised over rung `k` (H = nharms // k, M = 2H), the PRUNED width bank, and
    the boxcar's phase, then averaged over the sub-bin phase of the pulse -- which
    the search cannot choose and the model must not be allowed to.
    """
    ks = decimation_set(nharms, maxdecim)
    if hmax_data is None:
        hmax_data = nharms
    # **The phase grid must be SUB-BIN, and it must be one grid shared by every
    # rung.**  Sampling the pulse phase over a whole turn does nothing: the max
    # over the boxcar's integer start bin already absorbs whole-bin shifts, so 16
    # phases spread over [0, 1) are 16 copies of the same answer whenever the
    # shift is a whole number of bins -- which it is, for every rung, when the
    # grid is a divisor of the fold.  (That mistake made the deepest ladders read
    # ~9% high, because a pulse split across two bins never got sampled.)  The
    # grid below is `nsub` points per bin of the FINEST fold, spanning one whole
    # bin of the COARSEST, so each rung sees a uniform sweep of its own sub-bin
    # phase, and all rungs see the SAME pulse -- which is what lets the max over
    # rungs be taken before the average over phase, in that order.
    Mfine = 2 * (nharms // ks[0])
    nphase = int(nsub) * max(1, Mfine // (2 * (nharms // ks[-1])))
    phi = np.arange(nphase) / (Mfine * nsub)
    best = np.zeros(nphase)
    best_kw = [(ks[0], 1, Mfine)] * nphase
    for k in ks:
        H = nharms // k
        M = 2 * H
        nfill = int(min(H, hmax_data))
        if nfill < 1:
            continue
        h = np.arange(1, nfill + 1)
        ws = np.array(ladder_boxcar_widths(M, k, ks, fsp, maxfrac), dtype=int)
        D = _dirichlet(h[None, :], ws[:, None], M)                 # (nw, nfill)
        # var: the Nyquist harmonic of the profile enters at half weight, and
        # only if it is filled at all.
        v = 2.0 * np.abs(D) ** 2
        if nfill == H:
            v[:, -1] *= 0.25
        var = v.sum(axis=1)
        # signal, for every (sub-phase, width): one real inverse transform.
        X = np.zeros((nphase, len(ws), H + 1), dtype=complex)
        X[:, :, 1:nfill + 1] = (A[None, None, :nfill] * D[None, :, :]
                                * np.exp(2j * np.pi * h[None, None, :] * phi[:, None, None]))
        if nfill == H:
            X[:, :, H] = X[:, :, H].real
        S = np.fft.irfft(X, M, axis=2) * M                       # (nphase, nw, M)
        val = (S.max(axis=2) / np.sqrt(var)[None, :])            # (nphase, nw)
        iw = val.argmax(axis=1)
        top = val[np.arange(nphase), iw]
        upd = top > best
        best = np.where(upd, top, best)
        for j in np.nonzero(upd)[0]:
            best_kw[j] = (k, int(ws[iw[j]]), M)
    eff = float(best.mean())
    # Report the rung that wins most often, not the one that wins on one phase.
    from collections import Counter
    k, w, M = Counter(best_kw).most_common(1)[0][0]
    return eff, k, w, M


def efficiency_for(ducy, w10_w50, f0, dt, nharms=60, maxdecim=6, nsub=4,
                   nph=1 << 12, _cache={}):
    """`ladder_efficiency` straight from a drawn pulsar.  Builds the profile via
    `mc_profiles.make_profile`, so it is the SAME shape that was injected."""
    import mc_profiles as MP
    key = (round(float(ducy), 5), round(float(w10_w50), 4), nph)
    A = _cache.get(key)
    if A is None:
        prof, _ = MP.make_profile(ducy, w10_w50, nph=nph)
        A = profile_harmonics(prof, max(nharms, 4))
        if len(_cache) > 20000:
            _cache.clear()
        _cache[key] = A
    hmax = int(math.floor(0.5 / (dt * f0)))
    return ladder_efficiency(A, nharms, maxdecim, hmax, nsub)


# ---------------------------------------------------------------------------
# Section 5: prepfold's drizzle, and what it does to snr1
# ---------------------------------------------------------------------------
def fold_covariance(nbins, dpb, nrot=4096, _cache={}):
    """Exact autocovariance `C[0..d]` of a drizzled fold's profile bins, in units
    of the time-series sample variance: `C[d] = cov(bin_b, bin_{b+d})`.

    PRESTO's `fold()` spreads each sample's value across the profile bins its
    finite duration covers, in proportion to the overlap.  That makes the profile
    a LINEAR map of the samples, so its covariance is exact arithmetic on the
    weights -- no Monte Carlo and nothing fitted, which matters because
    `DOF_corr` itself is an empirical fit and there is no closed form to match.

    `dpb` is samples per profile bin (`P / nbins / dt`).  For `dpb >= 1` a sample
    touches at most two adjacent bins and `C` is tridiagonal, approaching
    `C[0] = dpb - 1/3`, `C[1] = 1/6` once averaged over sub-bin sample phase.
    Below 1 -- which prepfold does reach, at `dpb = 0.995`, because its bin count
    is a rounded quantity -- a sample spans several bins and `C` is wider; the
    general accumulation below covers both, so nothing has to special-case the
    boundary.
    """
    nbins = int(nbins)
    dpb = float(dpb)
    if dpb >= 1.0:
        # Closed form, and it is not an approximation worth worrying about: with
        # `dpb >= 1` a sample touches at most two bins, its sub-bin start phase is
        # equidistributed (the fold's period is not commensurate with `dt`), and
        # the two expectations integrate exactly.  Checked against the numeric
        # accumulation below at `dpb` = 1.04 ... 271: identical to 4 decimals.
        # It has to be closed form: `dpb` is continuous, so a cache keyed on it
        # never hits, and building a half-million-element weight array per
        # injection made the whole analysis 100x slower than reading the files.
        return np.array([dpb - 1.0 / 3.0, 1.0 / 6.0])
    key = (nbins, round(dpb, 4), int(nrot))
    if key in _cache:
        return _cache[key]
    L = 1.0 / dpb                       # sample duration, in profile bins
    nspan = int(math.ceil(L)) + 1
    n = int(nrot) * nbins
    s = (np.arange(n, dtype=float) * L) % nbins
    b0 = np.floor(s)
    W = np.empty((nspan, n))
    for j in range(nspan):
        lo = np.maximum(s, b0 + j)
        hi = np.minimum(s + L, b0 + j + 1.0)
        W[j] = np.clip(hi - lo, 0.0, None) / L
    # Per ROTATION, so this branch is on the same scale as the closed form above:
    # `n` samples cover `n / (nbins * dpb)` rotations of the fold.
    norm = nbins * (n / (nbins * dpb))
    C = np.array([float(np.sum(W[:nspan - d] * W[d:])) / norm
                  for d in range(nspan)])
    _cache[key] = C
    return C


def _boxcar_template(nbins, w):
    """riptide's zero-mean unit-L2 boxcar: `snr1 = sum_j h_j P_j / sigma_bin`."""
    n = int(nbins)
    a = math.sqrt((n - w) / (n * w))
    b = w / (n - w) * a
    h = np.full(n, -b)
    h[:w] = a
    return h


def drizzle_boxcar_corr(nbins, dpb, w, _cache={}):
    """Factor to MULTIPLY `snr1` by, for a profile from a drizzled fold.

    `snr1` divides by `sigma_bin * sqrt(w (1 - w/nbins))`, which is the sd of the
    boxcar sum only when the profile bins are independent.  Drizzling correlates
    adjacent bins positively, so the true sd is larger and `snr1` reads high.

    **It is width-dependent, and `sqrt(DOF_corr)` is only its wide-`w` limit.**  A
    one-bin boxcar cannot feel bin-to-bin correlation at all, so the factor is
    ~1.00 at `w = 1` and falls towards `sqrt(DOF_corr)` for wide ones.  Applying a
    flat `sqrt(DOF_corr)` instead over-corrects narrow pulses by ~20% at
    `dpb ~ 1`, which would trade prepfold's high-frequency inflation for a
    low-duty deflation -- along the study's main axis.
    """
    key = (int(nbins), round(float(dpb), 3), int(w))
    v = _cache.get(key)
    if v is not None:
        return v
    n, w = int(nbins), int(w)
    if w < 1 or w >= n:
        return 1.0
    C = fold_covariance(n, dpb)
    h = _boxcar_template(n, w)
    # var(sum_j h_j P_j) for a circulant covariance: sum_d C[d] * (h . roll(h, d)),
    # counting d != 0 twice.  `snr1` assumes it is C[0] * |h|^2 == C[0].
    var = C[0] * float(np.dot(h, h))
    for d in range(1, min(len(C), n)):
        var += 2.0 * C[d] * float(np.dot(h, np.roll(h, d)))
    v = math.sqrt(C[0] / var) if var > 0 else 1.0
    _cache[key] = v
    return v


def prepfold_sigma(nbins, dpb, nsamp_used=None):
    """Analytic per-bin noise sd of a prepfold profile on unit-variance data.

    `sqrt(C0 * dpb_actual)` -- i.e. the number of samples a bin receives, reduced
    by the fraction each shared sample gives away.  Section 8's `sqrt(N/nbins)` is
    this with the drizzle ignored; at `dpb ~ 1` that is 22% wrong, which is why
    the drizzle covariance is used here too.

    A 128-point MAD, which is what run 1 used, has ~9% sampling error on its own
    and reported S/N is exactly `1/sigma-hat`, so replacing it removes a scatter
    term comparable to everything else in prepfold's column.
    """
    C = fold_covariance(int(nbins), float(dpb))
    nrot = (nsamp_used / (nbins * dpb)) if nsamp_used else 1.0
    return math.sqrt(C[0] * max(nrot, 1e-9))
