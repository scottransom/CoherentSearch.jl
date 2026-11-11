using Test
using FFTW: FFTW
using CoherentSearch
using CoherentSearch: output_path, plot_stem, main, _plans!, Workspace, parse_cmdline
using Logging: with_logger, NullLogger
using Random: Random

# ---------------------------------------------------------------------------
# A small synthetic observation on disk, so the CLI tests need no external data.
# Amplitudes are flat noise-like values plus a bright harmonic series, which
# gives the search something to find (and hence something to write out).
# ---------------------------------------------------------------------------
function write_synthetic_fft(dir, name; N = 1 << 15, dt = 1.0e-4, bin0 = 210)
    nfreq = N ÷ 2
    amps = ComplexF32[ComplexF32(0.5f0, 0.5f0) for _ in 1:nfreq]
    for h in 1:8
        b = bin0 * h + 1
        b <= nfreq && (amps[b] = ComplexF32(60.0f0 / h, 0.0f0))
    end
    fftpath = joinpath(dir, name * ".fft")
    write(fftpath, amps)
    write(joinpath(dir, name * ".inf"),
          " Object being observed                      =  SYNTH\n" *
          " Epoch of observation (MJD)                 =  50000.0\n" *
          " Number of bins in the time series          =  $N\n" *
          " Width of each time series bin (sec)        =  $dt\n" *
          " Dispersion measure (cm-3 pc)               =  0.0\n")
    return fftpath, N * dt
end

@testset "output_path: candidate filename resolution" begin
    # One file, no -o, no --outdir: stdout, as it has always been.
    @test output_path("/data/obs_DM12.fft", "", "", 1) == ""
    # One file with -o: verbatim.
    @test output_path("/data/obs_DM12.fft", "cands.txt", "", 1) == "cands.txt"
    # Several files: .cohout beside each input, and -o must not reach here.
    @test output_path("/data/obs_DM12.fft", "", "", 3) == "/data/obs_DM12.cohout"
    # --outdir relocates the basename, and forces .cohout naming even for one file.
    @test output_path("/data/obs_DM12.fft", "", "/out", 1) == "/out/obs_DM12.cohout"
    @test output_path("/data/obs_DM12.fft", "", "/out", 3) == "/out/obs_DM12.cohout"
    # A name whose stem contains dots keeps them (DMs are written like DM445.0).
    @test output_path("/d/PM0063_DM445.0_red.fft", "", "", 2) ==
          "/d/PM0063_DM445.0_red.cohout"
end

@testset "plot_stem: PNG stem resolution" begin
    # An explicit --plotstem always wins.
    @test plot_stem("mystem", "cands.cohout", "x.fft") == "mystem"
    # A .cohout name loses only that extension.
    @test plot_stem("", "/d/obs_DM12.cohout", "/d/obs_DM12.fft") == "/d/obs_DM12_profiles"
    # A user-supplied -o name is used verbatim: a trailing token like `sd2_0.5`
    # looks like an extension but is part of the name, so it must not be stripped.
    @test plot_stem("", "cands_sd2_0.5", "x.fft") == "cands_sd2_0.5_profiles"
    @test plot_stem("", "cands.txt", "x.fft") == "cands.txt_profiles"
    # No output file (stdout run): fall back to the FFT basename.
    @test plot_stem("", "", "/data/obs_DM12.fft") == "obs_DM12_profiles"
end

mktempdir() do dir
    fftpath, T = write_synthetic_fft(dir, "synth")
    ft = FFTFile(fftpath)
    params = SearchParams(nharms = 8)
    kw = (lofreq = 200 / T, hifreq = 240 / T, blocksize = 64,
          threshold = 0.0, progress = :none, wisdom = false)

    @testset "SearchCache reuse leaves results unchanged" begin
        # The whole point of the cache is that a multi-file run reuses one file's
        # plans and workspaces for the next.  Reuse must be invisible in the
        # results: same candidates, bit-for-bit.
        fresh = search(ft, params; kw...)
        @test !isempty(fresh)

        cache = SearchCache()
        first_cached = search(ft, params; cache = cache, kw...)
        second_cached = search(ft, params; cache = cache, kw...)   # the reuse path
        for got in (first_cached, second_cached)
            @test length(got) == length(fresh)
            @test all(c.r === f.r && c.metric === f.metric && c.nharm == f.nharm
                      for (c, f) in zip(got, fresh))
        end

        # A second file (here, the same data reopened) must also match, since only
        # file-independent setup is cached.
        reopened = search(FFTFile(fftpath), params; cache = cache, kw...)
        @test all(c.metric === f.metric for (c, f) in zip(reopened, fresh))
    end

    @testset "measure_ducy fills the duty cycle without moving anything else" begin
        cands = search(ft, params; kw...)
        @test all(isnan(c.ducy) for c in cands)      # the hot loop does not measure it
        measured = measure_ducy(ft, cands, params)
        @test length(measured) == length(cands)
        for (m, c) in zip(measured, cands)
            # Only `ducy` changes; the detection itself must be untouched.
            @test m.freq === c.freq && m.metric === c.metric &&
                  m.r === c.r && m.nharm == c.nharm
            @test 0 < m.ducy <= params.boxcar_maxfrac
        end
        # The injected signal is a harmonic series with a narrow pulse, so the
        # strongest candidate should be matched by a narrow boxcar, not the widest.
        best = measured[argmax([c.metric for c in measured])]
        @test best.ducy < params.boxcar_maxfrac
    end

    @testset "SearchCache rebuilds when params or blocksize change" begin
        cache = SearchCache()
        search(ft, params; cache = cache, kw...)
        ws1 = cache.workspaces

        # Same params object, same Nprof: the very same objects come back.
        ws = _plans!(cache, params, 64, 1)
        @test ws === ws1

        # A different blocksize invalidates: Workspaces are sized by Nprof, and
        # reusing an undersized one would silently corrupt the chunk fill.
        ws2 = _plans!(cache, params, 128, 1)
        @test ws2 !== ws1
        @test cache.Nprof == 128

        # Different params, likewise.  Reuse is keyed on object identity, so an
        # equal-but-distinct SearchParams also rebuilds (conservative, correct).
        other = SearchParams(nharms = 8, hidr = 0.25)
        ws3 = _plans!(cache, other, 128, 1)
        @test ws3 !== ws2
        @test cache.params === other

        # More tasks than before tops the vector up rather than rebuilding.
        ws4 = _plans!(cache, other, 128, 3)
        @test length(ws4) >= 3
        @test all(w isa Workspace for w in ws4)
        # …and the extra workspaces persist for a later, smaller request.
        ws5 = _plans!(cache, other, 128, 1)
        @test ws5 === ws4
    end
end

@testset "prime_wisdom builds every plan the search uses" begin
    # `prime_wisdom` constructs a `Workspace` purely for its FFTW-planning side
    # effect, so it is the one caller that breaks silently when the Workspace
    # constructor's signature changes — which is exactly how it broke: nothing
    # exercised it, and it kept calling a `build_harmonic_plans` that had been
    # deleted.  Cheap to cover, so cover it.
    path = tempname()
    params = SearchParams(nharms = 8, decimations = decimation_set(8, 2))
    got = prime_wisdom(params; blocksize = 64, rigor = FFTW.MEASURE, path = path)
    @test got == path
    @test isfile(path) && filesize(path) > 0
    # And the wisdom it wrote is loadable, which is the whole point of writing it.
    @test import_wisdom!(path)
    rm(path; force = true)
end

@testset "prime_wisdom's default primes the plans a DEFAULT run actually builds" begin
    # The bug this pins (found 2026-08-22): `prime_wisdom`'s default was
    # `SearchParams()` — nharms=32, decimations=[1], i.e. ONE 64-bin transform —
    # while the CLI defaults to nharms=60, --maxdecim 6, i.e. six transforms of
    # 120/60/40/30/24/20 bins, five of them on strided views.  So the function
    # documented as "run once per host" primed a transform the search never
    # executes and none of the six it does.  The test above could not catch it
    # because it passes `params` explicitly, which is exactly the case that worked.
    a = parse_cmdline(["dummy.fft"])
    cli = SearchParams(nharms = a["nharms"], m = a["m"], hidr = a["hidr"],
                       threshold = a["threshold"],
                       decimations = decimation_set(a["nharms"], a["maxdecim"]),
                       precision = Symbol(a["precision"]))
    prod = production_params()
    @test prod.nharms == cli.nharms
    @test prod.decimations == cli.decimations

    # And the transform lengths really are the ones a Workspace builds: the base
    # brfft is 2*nharms bins, plus one per decimation factor at 2*fld(nharms,k).
    want = sort(unique(vcat(2 * cli.nharms,
                            [2 * fld(cli.nharms, k) for k in cli.decimations if k > 1])))
    ws = CoherentSearch.Workspace(prod, 64)
    got = sort(unique(vcat(size(ws.profs, 1), [size(d.dprofs, 1) for d in ws.decims])))
    @test got == want
    @test length(got) == 6            # 120/60/40/30/24/20 — not 1
end

@testset "main: multi-file run writes one .cohout per input" begin
    mktempdir() do dir
        p1, T = write_synthetic_fft(dir, "obs_DM10.0")
        p2, _ = write_synthetic_fft(dir, "obs_DM20.0"; bin0 = 214)
        argv = [p1, p2, "--noplot", "--noprogress", "--nowisdom",
                "--nharms", "8", "--blocksize", "64", "--threshold", "0.0",
                "--ncands", "3",
                "--lofreq", string(200 / T), "--hifreq", string(240 / T)]
        with_logger(NullLogger()) do
            main(argv)
        end

        o1 = joinpath(dir, "obs_DM10.0.cohout")
        o2 = joinpath(dir, "obs_DM20.0.cohout")
        @test isfile(o1) && isfile(o2)
        # Each file gets its *own* candidates — the second must not overwrite the
        # first, and the different injected bin must show up as a different freq.
        l1, l2 = readlines(o1), readlines(o2)
        @test length(l1) > 1 && length(l2) > 1
        @test startswith(l1[1], "#")
        f1 = parse(Float64, split(l1[2])[3])
        f2 = parse(Float64, split(l2[2])[3])
        @test !isapprox(f1, f2; rtol = 1e-3)

        # The Ducy(%) column: present in the header, and a real number for every
        # :boxcar candidate (the default metric).  It is the quantity compared
        # against riptide's `ducy`, so it must never silently come out blank.
        @test occursin("Ducy(%)", l1[1])
        for line in l1[2:end]
            f = split(line)
            @test length(f) == 6
            d = parse(Float64, f[6])
            # A duty cycle is a fraction of the profile, and the width bank tops
            # out at boxcar_maxfrac = 0.3 of it.
            @test 0 < d <= 30.0
        end

        # -o and --plotstem name a single output; with several inputs they would
        # have each file clobber the last, so they must be rejected outright.
        @test_throws ArgumentError main(vcat(argv, ["-o", joinpath(dir, "x.txt")]))
        @test_throws ArgumentError main(vcat(argv, ["--plotstem", "stem"]))
    end
end

@testset "main: single file keeps its historical behaviour" begin
    mktempdir() do dir
        p1, T = write_synthetic_fft(dir, "solo")
        base = [p1, "--noplot", "--noprogress", "--nowisdom", "--nharms", "8",
                "--blocksize", "64", "--threshold", "0.0", "--ncands", "3",
                "--lofreq", string(200 / T), "--hifreq", string(240 / T)]

        # No -o: candidates go to stdout and no file is created.
        # `redirect_stdout` needs a real stream, not an IOBuffer.
        captured = joinpath(dir, "stdout.txt")
        open(captured, "w") do io
            with_logger(NullLogger()) do
                redirect_stdout(io) do
                    main(base)
                end
            end
        end
        @test !isfile(joinpath(dir, "solo.cohout"))
        @test occursin("Frequency (Hz)", read(captured, String))

        # -o still writes exactly where it is told.
        named = joinpath(dir, "mycands.txt")
        with_logger(NullLogger()) do
            main(vcat(base, ["-o", named]))
        end
        @test isfile(named)

        # --outdir opts a single file into .cohout naming, creating the directory.
        od = joinpath(dir, "sub", "dir")
        with_logger(NullLogger()) do
            main(vcat(base, ["--outdir", od]))
        end
        @test isfile(joinpath(od, "solo.cohout"))
    end
end

# ---------------------------------------------------------------------------
# Backported from PRESTO's coherentsearch (November 2026): the normalisation
# check, the comb filter, the --deep pass and its sift against the default
# search's .cohout.
# ---------------------------------------------------------------------------

# Exponential powers of mean `scale^2` (Re, Im each of variance scale^2/2), plus
# optionally a pure sinusoid at integer bin `sinbin`.
function write_noise_fft(dir, name; N = 1 << 16, dt = 1.0e-3, scale = 1.0,
                         sinbin = 0, sinamp = 0.0, seed = 1)
    rng = Random.MersenneTwister(seed)
    nfreq = N ÷ 2
    amps = ComplexF32[scale * ComplexF32(randn(rng), randn(rng)) / sqrt(2f0)
                      for _ in 1:nfreq]
    sinbin > 0 && (amps[sinbin + 1] = ComplexF32(sinamp, 0))
    fftpath = joinpath(dir, name * ".fft")
    write(fftpath, amps)
    write(joinpath(dir, name * ".inf"),
          " Object being observed                      =  NOISE\n" *
          " Epoch of observation (MJD)                 =  50000.0\n" *
          " Number of bins in the time series          =  $N\n" *
          " Width of each time series bin (sec)        =  $dt\n" *
          " Dispersion measure (cm-3 pc)               =  0.0\n")
    return fftpath, N * dt
end

@testset "check_normalisation reads the noise scale from the file" begin
    mktempdir() do dir
        good, _ = write_noise_fft(dir, "good")
        @test (@test_logs check_normalisation(FFTFile(good)))      # silent
        # 5% off in the noise scale is inside the 10% tolerance...
        near, _ = write_noise_fft(dir, "near"; scale = 1.05)
        @test (@test_logs check_normalisation(FFTFile(near)))
        # ...20% is not, and nor (by a factor) is an un-normalised file.
        for (nm, sc) in (("hi", 1.2), ("raw", 1000.0))
            p, _ = write_noise_fft(dir, nm; scale = sc)
            @test !(@test_logs (:warn, r"do not look normalised") check_normalisation(FFTFile(p)))
        end
        # A short FFT falls back to its top half.
        short, _ = write_noise_fft(dir, "short"; N = 4096)
        @test (@test_logs check_normalisation(FFTFile(short)))
    end
end

@testset "remove_periodic drops a comb, keeps the fundamental" begin
    mktempdir() do dir
        B = 4000                                   # the sinusoid's Fourier bin
        p, T = write_noise_fft(dir, "comb"; sinbin = B, sinamp = 200.0)
        ft = FFTFile(p)
        mk(r, nh) = Candidate(r / T, 10.0, Float64(r), nh)
        # The sinusoid is harmonic 8 of r = B/8: a profile of eight identical peaks.
        comb = mk(B / 8, 16)
        fund = mk(B, 16)                           # the sinusoid itself, harmonic 1
        lownh = mk(B / 4, 6)                       # nharm below minharm: never judged
        h4 = mk(B / 4, 16)                         # strongest is harmonic 4 (< minharm)
        noise = mk(1234.5, 16)
        kept = remove_periodic(ft, [comb, fund, lownh, h4, noise])
        @test kept == [fund, lownh, h4, noise]     # order preserved
        @test remove_periodic(ft, [comb]; minharm = 9) == [comb]
        @test isempty(remove_periodic(ft, Candidate[]))
    end
end

@testset "remove_harmonics_of: only STRONGER related reference candidates" begin
    c(r, s) = Candidate(r / 100, s, r, 8)
    ref = [c(1000.0, 20.0), c(3333.3, 5.0)]
    deep = [c(500.0, 8.0),       # f/2 of a stronger ref: dropped
            c(1000.4, 9.0),      # identical (within tol) to a stronger ref: dropped
            c(1666.65, 9.0),     # f/2 of a WEAKER ref: kept
            c(777.7, 9.0),       # unrelated: kept
            c(250.0, 30.0)]      # related, but stronger than the ref: kept
    # numharm = 4 keeps the "unrelated" ones unrelated: at the default 16, 1666.65
    # is 5/3 of 1000 and 777.7 is ~9/7 of it, both within a bin.
    @test remove_harmonics_of(deep, ref; numharm = 4) == deep[3:5]
    @test remove_harmonics_of(deep, Candidate[]) == deep
end

@testset "read_cohout round-trips write_candidates, and reads PRESTO labels" begin
    mktempdir() do dir
        T = 1000.0
        cands = [Candidate(7.5, 12.34, 7500.0, 10, 0.065),
                 Candidate(0.26, 7.32, 260.0, 60)]
        f = joinpath(dir, "x.cohout")
        with_logger(NullLogger()) do
            CoherentSearch.write_candidates(cands, f, 6.0)
        end
        back = read_cohout(f, T)
        @test length(back) == 2
        @test [b.metric for b in back] ≈ [12.34, 7.32]
        @test [b.r for b in back] ≈ [7500.0, 260.0]
        @test [b.nharm for b in back] == [10, 60]
        @test back[1].ducy ≈ 0.065 && isnan(back[2].ducy)
        # A row PRESTO's coherentsearch labelled with a known pulsar.
        open(f, "a") do io
            println(io, "3        9.00      1.000000000000   1000.000000000000    16     5.00  J1234+5678(2)")
        end
        @test read_cohout(f, T)[3].freq == 1.0
    end
end

@testset "--deep: defaults, explicit overrides, naming and the default-search sift" begin
    a = parse_cmdline(["x.fft"])
    @test (a["nharms"], a["maxdecim"], a["hifreq"]) == (60, 6, 125.0)
    a = parse_cmdline(["x.fft", "--deep"])
    @test (a["nharms"], a["maxdecim"], a["hifreq"]) == (120, 12, 5.0)
    # Explicit options win, including one equal to the plain default.
    a = parse_cmdline(["x.fft", "--deep", "--nharms", "60", "--hifreq", "2"])
    @test (a["nharms"], a["maxdecim"], a["hifreq"]) == (60, 12, 2.0)
    # bin/prime_wisdom.jl primes the --deep plans too: they must be the ones a
    # --deep run builds (cf. the prime_wisdom-default bug pinned above).
    a = parse_cmdline(["x.fft", "--deep"])
    pd = production_params(deep = true)
    @test pd.nharms == a["nharms"]
    @test pd.decimations == decimation_set(a["nharms"], a["maxdecim"])
    @test output_path("/d/obs.fft", "", "", 2; deep = true) == "/d/obs_deep.cohout"
    @test output_path("/d/obs.fft", "", "/o", 1; deep = true) == "/o/obs_deep.cohout"

    mktempdir() do dir
        p, T = write_synthetic_fft(dir, "obs")
        common = ["--noprogress", "--nowisdom", "--blocksize", "64",
                  "--threshold", "0.0", "--outdir", dir, "--maxdecim", "1"]
        # Deep pass over the bright series' f/2 region with nothing to sift
        # against: its candidates survive.
        deepargs = vcat([p, "--deep", "--nharms", "8", "--ncands", "50",
                         "--lofreq", string(100 / T), "--hifreq", string(110 / T)], common)
        with_logger(NullLogger()) do
            main(deepargs)
        end
        deepout = joinpath(dir, "obs_deep.cohout")
        @test isfile(deepout) && !isfile(joinpath(dir, "obs.cohout"))
        before = read_cohout(deepout, T)
        @test any(c -> isapprox(c.r, 105; atol = 1), before)   # f/2 of bin 210
        # Now a default search finds the series at bin 210, far more strongly...
        with_logger(NullLogger()) do
            main(vcat([p, "--nharms", "8", "--ncands", "50",
                       "--lofreq", string(200 / T), "--hifreq", string(240 / T)], common))
        end
        @test isfile(joinpath(dir, "obs.cohout"))
        # ...and a re-run deep pass drops its subharmonic, unless --noharmremove.
        with_logger(NullLogger()) do
            main(deepargs)
        end
        after = read_cohout(deepout, T)
        @test !any(c -> isapprox(c.r, 105; atol = 1), after)
        @test length(after) < length(before)
        with_logger(NullLogger()) do
            main(vcat(deepargs, ["--noharmremove"]))
        end
        @test any(c -> isapprox(c.r, 105; atol = 1), read_cohout(deepout, T))
    end
end
