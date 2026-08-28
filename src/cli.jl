# Command-line front-end for CoherentSearch.jl.
# Mirrors the Python `coherent_search` console script.
#
# Usage:
#   julia --project=. -t auto bin/coherent_search.jl FILE.fft [FILE2.fft ...] [options]
#
# Use `-t auto` (or JULIA_NUM_THREADS) to enable multi-threaded searching.
#
# This lives *inside* the package rather than in `bin/` so that `main` and the
# ArgParse table are compiled into the precompile cache (see the workload in
# `CoherentSearch.jl`).  As a script in `bin/`, inferring and codegen'ing `main`
# alone cost ~4.7 s on every single run — more than the search itself.
# `bin/coherent_search.jl` is now a shim that calls `CoherentSearch.main(ARGS)`.

using ArgParse
using Printf: @printf, @sprintf

function parse_cmdline(argv)
    s = ArgParseSettings(
        prog = "coherent_search",
        description = "Search PRESTO-style FFT files for pulsations using coherent harmonic folding.",
        epilog = """
        The input FFT file MUST be normalized (Fourier powers with mean ~ 1); the
        S/N metric assumes unit-variance noise, so an un-normalized FFT produces
        meaningless (hugely inflated) S/N values.  Normalize an un-normalized FFT
        with PRESTO's `rednoise` routine, which also removes red noise.  The FFT
        should also be barycentered and have known RFI zapped.  The detection
        metric is the peak boxcar matched-filter S/N over a geometric bank of
        top-hat widths -- riptide's `snr1` statistic exactly, so S/N values are
        directly comparable with rseek's -- whose pure-noise distribution is
        analytic and flat across widths and harmonic decimations, so one
        --threshold means one false-alarm rate for every k.
        Near-identical candidates are collapsed by default (--noremove disables it),
        as are harmonically-related ones -- the f/2, 2f, 3f/2, ... family of a real
        signal (--noharmremove disables it, --numharm sets the max harmonic).
        HARMONIC DECIMATION (--maxdecim, default 6) also folds each fundamental at
        2..k times its frequency almost for free, re-using the interpolated
        harmonics and summing nharms/k of them; the reported harmonic count
        identifies which decimation found each candidate.  It is what sets the
        top of the searched band: --hifreq is the highest FUNDAMENTAL, and the
        defaults 125 Hz x 6 cover spin frequencies to 750 Hz -- past the 716 Hz
        of the fastest known pulsar -- in 120 down to 20 profile bins.
        A progress meter prints to stderr (--progressbar for a bar, --noprogress off).
        Pass `-t auto` to Julia for multi-threaded runs.

        SEVERAL FFT FILES may be given, and searching them in one invocation is
        strongly preferred to one invocation each: Julia's compilation of the
        search is paid once (~10 s) rather than per file, and the harmonic plans,
        FFTW plans and per-thread workspaces are built once and reused, so each
        extra file costs only its own search time.  Use it for a DM or beam sweep.

        OUTPUT: with a single file and no -o, candidates go to stdout (as before).
        With several files -o is ambiguous and rejected; each file's candidates are
        instead written beside it as <fftfile-without-.fft>.cohout.  --outdir
        redirects those .cohout files into one directory (and selects .cohout
        naming even for a single file).  `bin/sift_candidates.py` reads .cohout.

        PLOTTING is OFF by default and enabled with --plot, at which point it is
        deferred: all searches run first, and the plotting backend (CairoMakie) is
        loaded once at the very end to plot every file's candidates.  Off by
        default because it costs ~9 s to load plus first-call compilation, and
        because deferring it keeps every input's mmap live to the end of the run
        (50 large files is tens of GB of address space pinned at once) -- neither
        of which a pipeline wants.  Candidates can always be plotted afterwards
        from the .cohout files with bin/plot_candidates.jl.  --noplot is still
        accepted, and ignored, so existing scripts keep working.
        """,
    )
    @add_arg_table! s begin
        "fftfile"
            help = "PRESTO FFT file(s) to be searched.  Searching many in one run amortises Julia's start-up compilation over all of them."
            required = true
            nargs = '+'
            arg_type = String
        "--threshold", "-t"
            help = "S/N cutoff for picking candidates"
            arg_type = Float64
            default = 8.0
        "--outputfilenm", "-o"
            help = "Output filename to record candidates (single input file only; default: stdout)"
            arg_type = String
            default = ""
        "--outdir"
            help = "Write per-file <basename>.cohout candidate files into this directory (created if needed).  Implies .cohout naming even for a single input file; default is beside each input .fft"
            arg_type = String
            default = ""
        "--nharms", "-n"
            help = "Number of harmonics to sum. 60 is composite, so it decimates evenly by 2,3,4,5,6"
            arg_type = Int
            default = 60
        "--ncands"
            help = "Maximum number of candidates to return"
            arg_type = Int
            default = 100
        "--lobin"
            help = "Lowest frequency bin to search"
            arg_type = Int
            default = 100
        "--lofreq"
            help = "Lowest frequency (in Hz) to search"
            arg_type = Float64
            default = 0.1
        "--hifreq"
            help = "Highest FUNDAMENTAL frequency (in Hz) to search. Decimation carries coverage the rest of the way, to --hifreq * --maxdecim; the default 125 x 6 = 750 Hz clears the fastest known pulsar (716 Hz) with headroom"
            arg_type = Float64
            default = 125.0
        "--hidr"
            help = "Fourier bin resolution at highest harmonic"
            arg_type = Float64
            default = 0.5
        "--m"
            help = "Fourier bins summed by the interpolation kernel (even). Cost is linear in m; the signal-power loss is ~0.2/m averaged over sub-bin offset (1.3% at m=16, 0.63% at m=32), against ~6.5% already lost to the --hidr trial grid. Since the weights went Float32 the kernel is cheap enough that m=32 costs less than m=16 used to, so it is a reasonable choice when sensitivity matters more than wall clock"
            arg_type = Int
            default = 16
            range_tester = x -> x > 0 && iseven(x)
        "--precision"
            help = "Element type of the profile stage (interpolated harmonic amplitudes, the batched inverse FFT, and the folded profiles the metric reads): 'f32' (default) or 'f64'. Everything reported stays Float64; f32 halves the bandwidth of the bulk arrays and is faster at every thread count on both development hosts (1.21x at -t 1). Use f64 to reproduce a pre-2026-08-24 run, or when a candidate's S/N must be bit-comparable with the reference path"
            arg_type = String
            default = "f32"
            range_tester = x -> x in ("f64", "f32")
        "--sigma"
            help = "Noise scale for the S/N metric: 'analytic' (default) computes it in closed form from the input FFT being normalised -- sqrt(2*nlow + 0.5*nnyq), counting only the harmonics that carry data -- which is free where measuring it costs ~15% of the metric phases, and has no sampling error. 'measured' estimates it per chunk with a robust MAD instead: use it when the noise level varies with Fourier frequency (bad RFI, or a rednoise pass that did not take), which the closed form cannot see. A sanity check compares the two on a few chunks and warns if they disagree"
            arg_type = String
            default = "analytic"
            range_tester = x -> x in ("measured", "analytic")
        "--blocksize"
            help = "Trial fundamentals per parallel chunk (Nprof).  0 (the default) resolves per backend: 2048 on the CPU, 65536 on the GPU.  The two backends want values 32x apart, and on a GPU the wrong end of that range costs up to 5.55x, so sweep yours once with bench/gpu_search_report.jl"
            arg_type = Int
            default = 0
        "--maxdecim"
            help = "Max harmonic-decimation factor k: also search 2..k times each fundamental, folding nharms/k harmonics. 1 disables it"
            arg_type = Int
            default = 6
        "--drtol"
            help = "Fourier-bin tolerance for collapsing near-identical candidates"
            arg_type = Float64
            default = 1.0
        "--numharm"
            help = "Max harmonic number when removing harmonically-related candidates"
            arg_type = Int
            default = 16
        "--noremove"
            help = "Do not collapse near-identical (duplicate) candidates"
            action = :store_true
        "--noharmremove"
            help = "Do not collapse harmonically-related candidates (f/2, 2f, 3f/2, ...)"
            action = :store_true
        "--progressbar"
            help = "Show a progress bar instead of the default text percentage meter"
            action = :store_true
        "--noprogress"
            help = "Do not print a progress meter"
            action = :store_true
        "--verbose", "-v"
            help = "Report the trial grid, chunking and interpolation phase-cycle lengths before searching"
            action = :store_true
        "--metricstats"
            help = "Report per-block, per-decimation metric stats (min/median/mean/std/max) to help set --threshold; writes a full per-block table to <stem>_metricstats.txt"
            action = :store_true
        "--normalize"
            help = "Adaptive threshold: a first pass measures the per-(decimation,frequency) noise, then the metric is normalised to a significance before thresholding (--threshold is then in noise-sigma units, comparable across decimations). Doubles the runtime."
            action = :store_true
        "--nowisdom"
            help = "Do not import/export the FFTW plan-wisdom cache (planning is re-measured every run)"
            action = :store_true
        "--wisdomfile"
            help = "Path to the FFTW plan-wisdom cache (default: per-host file under the Julia depot, or \$COHERENT_WISDOM)"
            arg_type = String
            default = ""
        "--plot"
            help = "Plot the candidate pulse profiles. Off by default: it loads CairoMakie (~9 s) and holds every input's mmap live until the end of the run, neither of which a pipeline wants. Plot afterwards from the candidate files with bin/plot_candidates.jl"
            action = :store_true
        "--gpu"
            help = "Run the search on a CUDA GPU.  Loads CUDA.jl on demand (~6 s once per invocation, amortised across a multi-file run).  Requires --sigma analytic (the default); --normalize and --metricstats are CPU-only for now.  Candidates are NOT bit-identical to the CPU path -- the GPU sums the same kernel in a different order and agrees to ~2e-7 -- so compare candidate lists, not file bytes"
            action = :store_true
        "--noplot"
            help = "Accepted and ignored -- plotting is already off unless --plot is given. Kept so existing pipelines and scripts do not break"
            action = :store_true
        "--plotstem"
            help = "Output path stem for profile-plot PNGs (single input file only; default: derived from the candidate filename or the FFT name)"
            arg_type = String
            default = ""
        "--plotcols"
            help = "Profile-plot grid columns per page"
            arg_type = Int
            default = 3
        "--plotrows"
            help = "Profile-plot grid rows per page"
            arg_type = Int
            default = 5
    end
    return parse_args(argv, s)
end

"""
    main(argv) -> Nothing

Entry point for the `coherent_search` command line.  Searches every FFT file in
`argv`, writing each one's candidates as it finishes, then — only with `--plot` —
loads the plotting backend *once* and plots them all.
"""
function main(argv)
    a = parse_cmdline(argv)
    fftfiles = a["fftfile"]::Vector{String}
    nfiles = length(fftfiles)

    # `-o` and `--plotstem` name a single output; with several inputs they would
    # silently have each file overwrite the last.  Reject rather than guess.
    if nfiles > 1 && !isempty(a["outputfilenm"])
        throw(ArgumentError("-o/--outputfilenm names one output file but $nfiles FFT files were given; " *
                            "omit it (each file gets its own .cohout) or use --outdir"))
    end
    if nfiles > 1 && !isempty(a["plotstem"])
        throw(ArgumentError("--plotstem names one plot stem but $nfiles FFT files were given; " *
                            "omit it (each file's plots are named after its own .cohout)"))
    end
    # `--plotstem` only means anything with `--plot`; silently ignoring it would
    # leave someone waiting for PNGs that are never coming.
    if !a["plot"] && !isempty(a["plotstem"])
        @warn "--plotstem given but plotting is off; pass --plot to produce them"
    end

    outdir = a["outdir"]
    isempty(outdir) || mkpath(outdir)

    maxdecim = a["maxdecim"]
    nharms = a["nharms"]
    decimations = decimation_set(nharms, maxdecim)
    # One `SearchParams` for every file — `SearchCache` keys its reuse on this
    # object's identity, so it must not be rebuilt inside the loop.
    params = SearchParams(
        nharms = nharms,
        m = a["m"],
        hidr = a["hidr"],
        threshold = a["threshold"],
        decimations = decimations,
        precision = Symbol(a["precision"]),
        sigma = Symbol(a["sigma"]),
    )

    # `--gpu` loads CUDA.jl on demand.  It is a WEAK dependency, so it is not in
    # the manifest of a CPU-only install and `using CoherentSearch` never pays for
    # it; the cost lands here, once per invocation (~6 s), and a run over many
    # files amortises it to nothing -- which is the deployment mode that matters
    # (see docs/gpu_design.md §7.2).
    backend = CPUBackend()
    if a["gpu"]
        Symbol(a["sigma"]) === :analytic || error(
            "--gpu currently requires --sigma analytic (the default).  The measured " *
            "scale is a per-chunk MAD and has no device implementation yet; " *
            "docs/gpu_design.md stage 4.")
        a["normalize"] && error("--gpu does not support --normalize yet (it needs the " *
                                "two-pass per-(k,frequency) statistics); run on the CPU.")
        a["metricstats"] && error("--gpu does not support --metricstats yet; run on the CPU.")
        @info "Loading CUDA (once per invocation)"
        try
            @eval Main using CUDA
        catch e
            error("--gpu needs CUDA.jl installed in this environment: " *
                  "`using Pkg; Pkg.add(\"CUDA\")`.  ($(sprint(showerror, e)))")
        end
        backend = require_gpu()
        @info "GPU backend active" device=string(backend)
        # `--blocksize` used to default to 2048 on both backends -- the CPU's
        # tuned value, and wrong on every GPU measured.  The penalty grows with
        # the card: 1.37x (RTX A400), 1.65x (GTX 1080), 3.91x (RTX A4000),
        # **5.55x** (A100-SXM4-80GB).  On the A100 that turns a 5.94x win over a
        # 20-core Xeon into 1.07x, silently.
        #
        # We still do NOT derive it per device.  §4.11 fitted `0.5 * L2 / 2912 B`
        # to three cards and the A100 refutes it by a factor of 32 -- same 40 MB
        # of L2 as the RTX 4000 SFF Ada, opposite end of the range, because
        # occupancy on 108 SMs beats L2 residency.  What is defensible is a
        # single constant chosen for its WORST case: across all six measured
        # cards, 65536 is never worse than **1.22x** off that card's own optimum
        # (A400 1.00x, 1080 ~1.03x, A4000 1.05x, 2080 Super 1.08x, A100 1.14x,
        # Ada 1.22x), against 1.37-5.55x for 2048.  All but the Ada's were
        # measured before the download/scan overlap, which helps small chunks
        # most; the Ada's own figure moved 1.13x -> 1.22x for that reason, so
        # treat the bound as provisional.  It also fails safe on
        # memory: 65536 needs 0.18 GiB of workspace and fits on the smallest card
        # here, where 131072 does not.  docs/gpu_design.md §4.12.
        if a["blocksize"] == 0
            a["blocksize"] = GPU_DEFAULT_BLOCKSIZE
            @info "Using the GPU default --blocksize $(GPU_DEFAULT_BLOCKSIZE) " *
                  "(the CPU's 2048 costs 1.4-5.6x on the cards measured).  This is " *
                  "one constant, not a per-device rule: it is within 1.22x of the " *
                  "optimum on all six cards measured, and the optimum itself spans " *
                  "8192-262144.  For the last few percent sweep your card once with " *
                  "`julia --project=. bench/gpu_search_report.jl FILE.fft`; see the " *
                  "GPU section of the README."
        elseif a["blocksize"] <= 2048
            # An explicit small value is still a poor choice on a GPU however it
            # was arrived at, so this is correct rather than a false positive.
            @warn "--blocksize $(a["blocksize"]) is tuned for the CPU and is a poor " *
                  "choice on every GPU measured (up to 5.55x off the best on an " *
                  "A100).  Omit --blocksize to get the GPU default of " *
                  "$(GPU_DEFAULT_BLOCKSIZE), or sweep your card with " *
                  "`julia --project=. bench/gpu_search_report.jl FILE.fft`."
        end
    end
    # The CPU keeps its own tuned value; the two backends differ by 32x and
    # nothing about the GPU finding above applies to `_search_region!`, where
    # shrinking the chunk was measured WORSE at every thread count (CLAUDE.md).
    a["blocksize"] == 0 && (a["blocksize"] = CPU_DEFAULT_BLOCKSIZE)

    cache = SearchCache()
    # Deferred plotting: (FFTFile, candidates, stem) per file with something to
    # show.  Holding the `FFTFile`s costs only virtual address space (they are
    # mmaps) and saves reopening them after the searches.
    toplot = Tuple{FFTFile,Vector{Candidate},String}[]

    for (i, path) in enumerate(fftfiles)
        nfiles > 1 && @info "File $i of $nfiles" file=path
        ft = FFTFile(path)
        outfile = output_path(path, a["outputfilenm"], outdir, nfiles)
        cands = search_one(ft, params, a, cache, backend)
        write_candidates(cands, outfile, a["threshold"])
        if a["plot"] && !isempty(cands)
            push!(toplot, (ft, cands, plot_stem(a["plotstem"], outfile, path)))
        end
    end

    # Hand the GPU's cross-file workspace back to the driver.  The device cache
    # is keyed on `(params, Nprof, r_lo)` and deliberately survives the loop
    # above -- rebuilding it per file cost ~0.5 s each, dominated by the
    # `CUDA.reclaim()` that used to run per file (docs/gpu_design.md §4.6).  A no-op
    # on the CPU backend.  In a `try`/`finally` so an error mid-batch still
    # releases: this is the only thing that returns device memory to the driver,
    # and on a display GPU that memory is the desktop's.
    try
        plot_all(toplot, params, a)
    finally
        release_backend!(backend)
    end
    return nothing
end

"""
    output_path(fftfile, outputfilenm, outdir, nfiles) -> String

Resolve where one file's candidates go.  A lone input file with neither `-o` nor
`--outdir` keeps the historical behaviour of writing to stdout (signalled by an
empty string).  Otherwise the name is `<fftfile stripped of .fft>.cohout`, beside
the input or in `outdir`.
"""
function output_path(fftfile, outputfilenm, outdir, nfiles)
    if nfiles == 1 && isempty(outdir)
        return outputfilenm          # "" ⇒ stdout, as before
    end
    stem = first(splitext(fftfile))  # keeps the directory of the input
    isempty(outdir) || (stem = joinpath(outdir, basename(stem)))
    return string(stem, ".cohout")
end

"""
    search_one(ft, params, a, cache) -> Vector{Candidate}

Search one FFT file with the parsed options `a`, reusing `cache`'s plans and
workspaces, and return its top `--ncands` candidates best-metric first.
"""
function search_one(ft::FFTFile, params::SearchParams, a, cache::SearchCache,
                    backend::SearchBackend = CPUBackend())
    @info "Searching" file=ft.path T=ft.T nharms=params.nharms decimations=params.decimations threads=nthreads() backend=nameof(typeof(backend))

    progress = a["noprogress"] ? :none : (a["progressbar"] ? :bar : :text)
    mstats = a["metricstats"] ? MetricStats() : nothing
    # `--gpu` loads CUDA (and hence the extension's `_region!` method) from
    # inside `main`, so that method is NEWER than the running function's world
    # age and a direct call raises "method too new to be called from this world
    # context".  Routing through `invokelatest` is the standard fix; it is only
    # needed on the GPU path, and only once per file, so it costs nothing.
    _dosearch() = search(ft, params;
                   lofreq = a["lofreq"], hifreq = a["hifreq"], lobin = a["lobin"],
                   blocksize = a["blocksize"], threshold = a["threshold"],
                   remove = !a["noremove"], dr_tol = a["drtol"],
                   harm_remove = !a["noharmremove"], numharm = a["numharm"],
                   progress = progress, metricstats = mstats,
                   normalize = a["normalize"], verbose = a["verbose"],
                   wisdom = !a["nowisdom"],
                   wisdom_file = isempty(a["wisdomfile"]) ? nothing : a["wisdomfile"],
                   cache = cache, backend = backend)
    cands = backend isa CPUBackend ? _dosearch() : Base.invokelatest(_dosearch)

    if mstats !== nothing
        # Per-file stem, so a multi-file run does not overwrite one file's tables
        # with the next one's.
        base = isempty(a["outputfilenm"]) ?
               first(splitext(basename(ft.path))) : a["outputfilenm"]
        report_metricstats(mstats, base, params, a["threshold"])
    end

    # Report the strongest `ncands` candidates, sorted best metric first.
    sort!(cands; by = c -> c.metric, rev = true)
    if length(cands) > a["ncands"]
        cands = cands[1:a["ncands"]]
    end
    # Measure each reported candidate's best-fitting boxcar duty cycle — only
    # affordable here, on the reported handful (see `measure_ducy`).
    if !isempty(cands)
        cands = measure_ducy(ft, cands, params)
    end
    return cands
end

"""
    write_candidates(cands, outfile, threshold)

Write the candidate table to `outfile`, or to stdout when `outfile` is empty.
"""
function write_candidates(cands::Vector{Candidate}, outfile::AbstractString, threshold)
    # Fixed-width columns; %.12g keeps at least 12 significant figures for the
    # frequency and period at any magnitude (fast pulsars have very short periods,
    # where a fixed number of decimal places would lose precision).
    # `Ducy(%)` is the best-fitting boxcar's duty cycle, defined as riptide's
    # `rseek` defines it (width / profile bins) so the two are directly
    # comparable; `-` means it was never measured (see `measure_ducy`).
    # The leading column is the 1-based RANK, and the header must say so: it did
    # not, and awk-ing `$2` for the frequency (it is the S/N) is then an easy and
    # silent mistake to make -- one that produced a confident, entirely wrong
    # reading of a real 220-file search on 2026-08-25.  `#Num` occupies exactly
    # the `%-4d` the rank is printed in, so the columns still line up.
    header = ["#Num    'S/N'      Frequency (Hz)        Period (ms)    #Harm  Ducy(%)"]
    lines = [@sprintf("%-4d  %8.2f  %18.12f  %18.12f   %3d   %6s",
                      i, c.metric, c.freq, 1000.0 / c.freq, c.nharm,
                      isnan(c.ducy) ? "-" : @sprintf("%.2f", 100 * c.ducy))
             for (i, c) in enumerate(cands)]
    outlines = vcat(header, lines)
    if isempty(outfile)
        foreach(println, outlines)
        println(stderr, "# $(length(cands)) candidates above threshold $(threshold)")
    else
        open(outfile, "w") do io
            foreach(l -> println(io, l), outlines)
        end
        @info "Wrote candidates" n=length(cands) file=outfile
    end
    return nothing
end

"""
    plot_all(toplot, params, a)

Plot every searched file's candidates in one pass.  The plotting backend
(CairoMakie) is loaded lazily *here*, after all searching is done, so ordinary
searches, tests and the cross-validation never pay for it — and a multi-file run
pays its ~9 s load once rather than once per file.
"""
function plot_all(toplot::Vector{Tuple{FFTFile,Vector{Candidate},String}}, params::SearchParams, a)
    isempty(toplot) && return nothing
    @info "Loading the plotting backend (CairoMakie) for --plot" files=length(toplot)
    # Include into `Main`, not into this module: `plotting.jl` depends on
    # CairoMakie and Dates, which belong to the *active project*, not to
    # CoherentSearch's own dependencies.  Including it here would resolve those
    # `using`s against this package and fail.
    Base.include(Main, joinpath(dirname(@__DIR__), "bin", "plotting.jl"))
    for (ft, cands, stem) in toplot
        # `include` defines CandidatePlots in a newer world age than this running
        # function.  Resolve the binding *and* call it inside an `invokelatest`
        # closure so both happen in the latest world (Julia 1.12 world semantics).
        files = Base.invokelatest() do
            Main.CandidatePlots.plot_candidates(ft, cands, params;
                                                outstem = stem,
                                                ncols = a["plotcols"], nrows = a["plotrows"])
        end
        @info "Wrote candidate profile plots" pages=length(files) stem=stem
    end
    return nothing
end

"""
    plot_stem(plotstem, outfile, fftfile) -> String

Resolve the PNG output stem: an explicit `--plotstem` wins; otherwise derive it
from the candidate output filename, or (for stdout runs) from the FFT filename.
"""
function plot_stem(plotstem, outfile, fftfile)
    isempty(plotstem) || return plotstem
    # Use the candidate filename verbatim except for a `.cohout` extension; do
    # NOT run splitext unconditionally, since a trailing token like `sd2_0.5`
    # looks like an extension but is part of a user-supplied `-o` name.
    if !isempty(outfile)
        stem, ext = splitext(outfile)
        return string(ext == ".cohout" ? stem : outfile, "_profiles")
    end
    return string(first(splitext(basename(fftfile))), "_profiles")
end

"""
    report_metricstats(ms, base, params, threshold)

Summarise the metric distribution collected in `ms::MetricStats`.  Prints to
`stderr` (1) a band-wide per-decimation table — exact moments plus the empirical
per-`k` metric value at a range of single-trial false-alarm probabilities — and
(2) how the FAP=1e-4 threshold *drifts across frequency* within each `k`, which
is where the red-noise (low `f`) and Nyquist-rolloff (high `f`) effects show up.
Writes three files for offline plotting / threshold fitting: the per-block table
`<base>_metricstats.txt`, the per-`(k, frequency window)` false-alarm thresholds
`<base>_metricfap.txt`, and the per-`(k, window)` histograms `<base>_metrichist.txt`.
"""
function report_metricstats(ms::MetricStats, base::String, params, threshold)
    if isempty(ms.hists) || all(h -> h.total == 0, ms.hists)
        @warn "No metric statistics collected (no blocks searched)"
        return
    end
    faps = (0.1, 0.01, 1e-3, 1e-4, 1e-5)
    faplabels = ("1e-1", "1e-2", "1e-3", "1e-4", "1e-5")
    summ = metricstats_summary(ms; faps=faps)
    wins = metricstats_windows(ms; faps=faps)

    # --- band-wide per-decimation summary to stderr ---
    println(stderr)
    println(stderr, "Metric statistics by decimation  (peak boxcar S/N, threshold=$(threshold))")
    println(stderr, "  Pure-noise floor scales ~sqrt(nbins); a fixed threshold picks a DIFFERENT false-alarm rate per k.")
    @printf(stderr, "  %-3s %4s %5s %10s %7s %7s %7s | metric at single-trial FAP =\n",
            "k", "Hk", "nbins", "ntrials", "mean", "std", "max")
    @printf(stderr, "  %-3s %4s %5s %10s %7s %7s %7s | %8s %8s %8s %8s %8s\n",
            "", "", "", "", "", "", "", faplabels...)
    for r in summ
        @printf(stderr, "  %-3d %4d %5d %10d %7.3f %7.3f %7.3f |",
                r.k, r.Hk, r.nbins, r.ntrials, r.mean, r.std, r.max)
        for v in r.fap
            @printf(stderr, " %8.3f", v)
        end
        r.overflow > minimum(faps) && @printf(stderr, "  (overflow %.1e!)", r.overflow)
        println(stderr)
    end
    println(stderr, "  A single --threshold gives these per-k FAPs; for comparable FAP, threshold each k separately")
    println(stderr, "  (or normalise the metric per (k, frequency)).  FAP columns are single-trial, single-decimation.")

    # --- frequency drift of the FAP=1e-4 threshold, per k, to stderr ---
    fi = findfirst(==(1e-4), faps)
    println(stderr)
    println(stderr, "  FAP=1e-4 threshold vs frequency ($(ms.nwin) log-spaced windows per k; drift = red noise / Nyquist):")
    @printf(stderr, "  %-3s %5s %21s %9s %9s\n", "k", "nbins", "freq range (Hz)", "min", "max")
    for k in sort(unique(r.k for r in wins))
        rk = [r for r in wins if r.k == k]
        isempty(rk) && continue
        vals = [r.fap[fi] for r in rk]
        @printf(stderr, "  %-3d %5d %9.3f -%9.3f %9.3f %9.3f\n",
                k, rk[1].nbins, minimum(r.flo for r in rk), maximum(r.fhi for r in rk),
                minimum(vals), maximum(vals))
    end
    println(stderr, "  Full per-window thresholds -> $(base)_metricfap.txt")
    println(stderr)

    # --- full per-block table ---
    blockfile = string(base, "_metricstats.txt")
    open(blockfile, "w") do io
        println(io, "# Per-block, per-decimation metric statistics")
        println(io, "# metric=boxcar nharms=$(params.nharms) threshold=$(threshold)")
        println(io, "# nbins = 2*Hk; the pure-noise metric floor scales ~sqrt(nbins).")
        @printf(io, "#%-7s %3s %4s %5s %10s %14s %14s %8s %9s %9s %9s %9s %9s\n",
                "block", "k", "Hk", "nbins", "ngoodbins", "f_lo(Hz)", "f_hi(Hz)",
                "n", "min", "median", "mean", "std", "max")
        for s in ms.blocks
            @printf(io, "%-8d %3d %4d %5d %10.3f %14.8f %14.8f %8d %9.3f %9.3f %9.3f %9.3f %9.3f\n",
                    s.block, s.k, s.Hk, s.nbins, s.ngoodbins, s.flo, s.fhi,
                    s.n, s.min, s.median, s.mean, s.std, s.max)
        end
    end

    # --- per-(k, window) false-alarm thresholds ---
    fapfile = string(base, "_metricfap.txt")
    open(fapfile, "w") do io
        println(io, "# Per-(decimation, frequency-window) empirical false-alarm thresholds")
        println(io, "# metric=boxcar nharms=$(params.nharms)")
        println(io, "# fap_X = metric value whose single-trial, single-decimation false-alarm prob is X (in-window)")
        @printf(io, "#%-3s %4s %5s %4s %13s %13s %10s %8s %8s %8s | %s\n",
                "k", "Hk", "nbins", "win", "f_lo(Hz)", "f_hi(Hz)", "ntrials",
                "mean", "std", "median", join(("fap" .* faplabels), " "))
        for r in wins
            @printf(io, "%-4d %4d %5d %4d %13.7f %13.7f %10d %8.3f %8.3f %8.3f |",
                    r.k, r.Hk, r.nbins, r.win, r.flo, r.fhi, r.ntrials, r.mean, r.std, r.median)
            for v in r.fap
                @printf(io, " %7.3f", v)
            end
            r.overflow > minimum(faps) && @printf(io, "  # overflow %.1e", r.overflow)
            println(io)
        end
    end

    # --- per-(k, window) histograms (only bins spanning the populated range) ---
    histfile = string(base, "_metrichist.txt")
    open(histfile, "w") do io
        println(io, "# Per-(decimation, frequency-window) metric histograms (one streaming pass)")
        println(io, "# metric=boxcar nharms=$(params.nharms)")
        println(io, "# bin = left edge of a bin of width $((ms.hist_hi - ms.hist_lo) / ms.hist_nb); count = trials in [bin, bin+width)")
        @printf(io, "#%-3s %4s %4s %13s %13s %12s %14s\n",
                "k", "Hk", "win", "f_lo(Hz)", "f_hi(Hz)", "bin", "count")
        w = (ms.hist_hi - ms.hist_lo) / ms.hist_nb
        for h in ms.whists
            firstnz = findfirst(>(0), h.counts)   # trim to populated range
            lastnz = findlast(>(0), h.counts)
            firstnz === nothing && continue
            for i in firstnz:lastnz
                @printf(io, "%-4d %4d %4d %13.7f %13.7f %12.5f %14d\n",
                        h.k, h.Hk, h.win, h.flo, h.fhi, ms.hist_lo + (i - 1) * w, h.counts[i])
            end
        end
    end

    @info "Wrote metric statistics" perblock=blockfile fap=fapfile histogram=histfile nblocks=length(ms.blocks)
    return
end
