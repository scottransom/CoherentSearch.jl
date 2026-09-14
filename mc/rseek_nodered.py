#!/usr/bin/env python3
"""`rseek`, with riptide's OWN time-domain dereddening switched off.

THE POINT.  The `rseek_W` arm asks one question: is riptide's collapse under red
noise its PREPROCESSING or its FFA?  It searches a time series we have already
whitened in the Fourier domain (`realfft` -> `rednoise` -> `realfft -inv`).  But
`rseek` always subtracts its own 4-second running median first, and running that
on an already-whitened series does not answer the question -- it is a second,
redundant high-pass that eats signal at long periods and leaves the comparison
measuring two cleanings instead of one.

`riptide.ffa_search` already takes `deredden=False`; it is only `rseek`'s command
line that does not expose it.  So this shim builds the arguments with riptide's
OWN parser, rebinds `ffa_search` in the app's namespace with `deredden` forced
off, and calls riptide's own `run_program`.  Everything else -- the peak finding,
the frequency clustering, the printed table -- is riptide's code, unmodified, so
the output cannot drift from `rseek`'s and `parse_rseek` keeps working.

**It deliberately does NOT patch the installed riptide.**  The package in use is
a site-packages copy (`0.2.7.dev1+g86aea2d74`), not an editable install of
`../riptide`, so editing the checkout would either do nothing or -- if
reinstalled -- change the binary that `rseek_A` and `rseek_B` are being measured
with, in the middle of the study.  A shim leaves those two arms bit-for-bit the
code they have always been.

Normalisation is KEPT.  `ffa_search` does `deredden` then `normalise`, and only
the first is switched off: riptide still scales the series to unit variance,
which it needs, and which the inverse FFT of a `rednoise`-normalised spectrum
does not provide (it comes back with a standard deviation of ~1e-3).
"""

import sys
from functools import partial

import riptide.apps.rseek as R


def main(argv=None):
    args = R.get_parser().parse_args(argv)
    # run_program looks `ffa_search` up in this module's globals at call time.
    R.ffa_search = partial(R.ffa_search, deredden=False)
    R.run_program(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
