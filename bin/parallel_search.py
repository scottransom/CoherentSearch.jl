#!/usr/bin/env python3
"""Run one coherent search over many .fft files as N independent parallel jobs.

    bin/parallel_search.py [-j N] [--gpu] [OPTIONS] FILE.fft ...  [-- SEARCH ARGS]

The search itself is single-threaded-per-process on the CPU (that is the
deployment model `docs/Summary_and_Future_Work.md` §3.1 argues for) and
single-device-per-process on the GPU (there is no multi-GPU support inside
`coherent_search.jl`).  Either way the way to use a whole machine is several
independent invocations over disjoint slices of the file list, and doing that by
hand is tedious and easy to get wrong.  This does it.

Two things it is careful about, both of which bite when you try this with GNU
parallel:

  * **One invocation per job, not one per file.**  Start-up is ~2.4 s on the CPU
    and ~6 s more on the GPU (loading CUDA), against ~1.1 s of marginal cost for
    an extra file in the same process -- and on the GPU the device workspace,
    the cuFFT plans and the pinned host buffers are all cached across files.
    `parallel ::: *.fft` throws all of that away on every file.  So the file list
    is PARTITIONED and each job gets its whole slice in one command line.

  * **A job that receives exactly one file would write to stdout.**  That is
    `coherent_search.jl`'s documented single-file behaviour, and it silently
    turns into a lost candidate list here.  When it can happen and no --outdir
    was given, this script supplies one per job (the input's own directory), so
    every file gets its `<stem>.cohout` beside it exactly as a multi-file run
    would.

Balance: files are dealt to the least-loaded job by SIZE (largest first), so a
heterogeneous glob does not leave one job running long after the others are
done.  For a DM sweep, where every file is the same size, that is the same as
dealing round-robin.  --split block keeps each job's slice contiguous instead.

Examples
--------
  # 6 GPUs, 220 DMs, one job per card
  bin/parallel_search.py --gpu -j 6 NGC6624_*_red.fft -- --blocksize 1048576

  # a whole CPU socket, one single-threaded process per core
  bin/parallel_search.py -j 20 *.fft -- --threshold 8

  # resume an interrupted run
  bin/parallel_search.py --gpu -j 6 --skip-existing --outdir cands *.fft
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEARCH = REPO / "bin" / "coherent_search.jl"


def visible_gpus():
    """Device ordinals this process may use, honouring CUDA_VISIBLE_DEVICES."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [d.strip() for d in env.split(",") if d.strip() != ""]
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        out = subprocess.run([smi, "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def partition(files, njobs, mode):
    """Split `files` into `njobs` lists.  Never returns an empty job."""
    njobs = max(1, min(njobs, len(files)))
    if mode == "block":
        # Contiguous slices, sizes differing by at most one.
        out, start = [], 0
        for j in range(njobs):
            n = len(files) // njobs + (1 if j < len(files) % njobs else 0)
            out.append(files[start:start + n])
            start += n
        return out
    # Longest-processing-time-first on file size: the standard greedy bound, and
    # for equal-sized files it degenerates to round-robin.
    sized = sorted(files, key=lambda f: -Path(f).stat().st_size)
    out = [[] for _ in range(njobs)]
    load = [0] * njobs
    for f in sized:
        j = load.index(min(load))
        out[j].append(f)
        load[j] += Path(f).stat().st_size
    return [sorted(g) for g in out]


def cohout_for(fft, outdir):
    stem = Path(fft).with_suffix("")
    return Path(outdir) / (stem.name + ".cohout") if outdir else stem.with_suffix(".cohout")


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Anything after `--` is passed through to bin/coherent_search.jl unchanged.")
    p.add_argument("files", nargs="+", help=".fft files (shell globs are fine)")
    p.add_argument("-j", "--jobs", type=int, default=0,
                   help="concurrent jobs; default = one per GPU with --gpu, else os.cpu_count()")
    p.add_argument("-t", "--threads", type=int, default=1,
                   help="Julia threads PER JOB (default 1: -j jobs x -t threads should not "
                        "exceed your cores)")
    p.add_argument("--gpu", action="store_true",
                   help="run each job on its own CUDA device via CUDA_VISIBLE_DEVICES, and "
                        "pass --gpu to the search")
    p.add_argument("--gpus", default="",
                   help="comma-separated device ordinals to use (default: all visible)")
    p.add_argument("--outdir", default="",
                   help="passed to the search; also where --skip-existing looks")
    p.add_argument("--split", choices=("size", "block"), default="size",
                   help="size (default): balance total bytes per job.  block: contiguous slices")
    p.add_argument("--skip-existing", action="store_true",
                   help="drop inputs whose .cohout already exists (resume a partial run)")
    p.add_argument("--julia", default="julia", help="julia executable")
    p.add_argument("--project", default=str(REPO), help="--project for julia")
    p.add_argument("--sysimage", default="", help="--sysimage for julia")
    p.add_argument("--logdir", default="", help="per-job logs (default: a temp dir, path printed)")
    p.add_argument("-n", "--dry-run", action="store_true", help="print the commands and exit")
    # Split on the first bare `--` OURSELVES.  With `nargs="+"` argparse hands
    # everything after it to `files` instead, so `-- --blocksize 8192` would be
    # warned about as a missing file and then silently dropped.
    argv = sys.argv[1:]
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    else:
        passthrough = []
    args = p.parse_args(argv)

    files = [f for f in args.files if Path(f).is_file()]
    missing = sorted(set(args.files) - set(files))
    if missing:
        print(f"warning: skipping {len(missing)} path(s) that are not files, "
              f"e.g. {missing[0]}", file=sys.stderr)
    if not files:
        sys.exit("no input .fft files")

    if args.skip_existing:
        before = len(files)
        files = [f for f in files if not cohout_for(f, args.outdir).exists()]
        print(f"--skip-existing: {before - len(files)} of {before} already done, "
              f"{len(files)} to run")
        if not files:
            return 0

    devices = []
    if args.gpu:
        devices = ([d.strip() for d in args.gpus.split(",") if d.strip()]
                   if args.gpus else visible_gpus())
        if not devices:
            sys.exit("--gpu given but no CUDA devices found (try --gpus 0)")

    njobs = args.jobs or (len(devices) if args.gpu else (os.cpu_count() or 1))
    groups = partition(files, njobs, args.split)
    njobs = len(groups)

    # Created here rather than left to N racing `mkpath` calls in the jobs.
    args.outdir and Path(args.outdir).mkdir(parents=True, exist_ok=True)

    logdir = Path(args.logdir) if args.logdir else Path(f".parallel_search.{os.getpid()}")
    logdir.mkdir(parents=True, exist_ok=True)

    procs, logs = [], []
    for j, group in enumerate(groups):
        cmd = [args.julia, f"--project={args.project}", f"-t{args.threads}"]
        if args.sysimage:
            cmd.append(f"--sysimage={args.sysimage}")
        cmd.append(str(SEARCH))
        if args.gpu:
            cmd.append("--gpu")
        outdir = args.outdir
        # See the module docstring: a one-file job with no --outdir writes to
        # stdout instead of a .cohout, which is not what the user asked for.
        if not outdir and len(group) == 1:
            outdir = str(Path(group[0]).parent or ".")
        if outdir:
            cmd += ["--outdir", outdir]
        # The per-chunk meter is noise in a log file nobody watches live, and it
        # is what the status line below replaces.  A user who explicitly asks for
        # one in the passthrough gets it.
        if not any(a.startswith("--progress") for a in passthrough):
            cmd.append("--noprogress")
        cmd += passthrough + group

        dev = devices[j % len(devices)] if args.gpu else None
        if args.dry_run:
            pre = f"CUDA_VISIBLE_DEVICES={dev} " if dev is not None else ""
            print(f"# job {j}: {len(group)} files")
            print(pre + " ".join(cmd))
            continue

        env = dict(os.environ)
        if dev is not None:
            env["CUDA_VISIBLE_DEVICES"] = dev
        log = logdir / f"job{j}.log"
        fh = open(log, "w")
        logs.append((log, fh, len(group)))
        procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env))

    if args.dry_run:
        return 0

    where = "device " + ",".join(devices) if args.gpu else f"{args.threads} thread(s) each"
    print(f"{njobs} jobs over {len(files)} files ({where}); logs in {logdir}")
    t0 = time.time()
    tty = sys.stdout.isatty()
    # `Info: Searching` is the per-file @info header, one per file.  Matching
    # bare "Searching" also catches the per-chunk progress meter ("Searching:
    #  1% (4/615 chunks)"), which reported 828/3 the first time this was run.
    pat = re.compile(r"Info: Searching")
    last = -1
    try:
        while any(pr.poll() is None for pr in procs):
            time.sleep(2.0)
            done = 0
            for log, fh, _ in logs:
                fh.flush()
                try:
                    done += len(pat.findall(log.read_text(errors="replace")))
                except OSError:
                    pass
            # A file is only finished when the NEXT one starts, so the count of
            # "Searching" lines is files started; report it as such.
            if done != last:
                el = time.time() - t0
                msg = f"  {done}/{len(files)} started   {el:6.1f} s elapsed"
                # `\r` only makes sense on a terminal; piped to a file or a log
                # it concatenates every update onto one unreadable line.
                print(("\r" + msg) if tty else msg, end="" if tty else "\n", flush=True)
                last = done
    except KeyboardInterrupt:
        print("\ninterrupted; terminating jobs")
        for pr in procs:
            pr.terminate()
    codes = [pr.wait() for pr in procs]
    for _, fh, _ in logs:
        fh.close()
    print(("\r" if tty else "") + f"done: {len(files)} files in {time.time() - t0:.1f} s"
          + " " * 20)

    bad = [j for j, c in enumerate(codes) if c != 0]
    if bad:
        for j in bad:
            print(f"job {j} exited {codes[j]} -- see {logs[j][0]}", file=sys.stderr)
            print("  " + "\n  ".join((logs[j][0].read_text(errors="replace")
                                      .splitlines() or ["<empty log>"])[-8:]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
