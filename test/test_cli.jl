using Test
using CoherentSearch
using CoherentSearch: output_path, plot_stem, main, _plans!, Workspace
using Logging: with_logger, NullLogger

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
    params = SearchParams(nharms = 8, metric = :boxcar)
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
        hp1, ws1 = cache.hplans, cache.workspaces

        # Same params object, same Nprof: the very same objects come back.
        hp, ws = _plans!(cache, params, 64, 1)
        @test hp === hp1 && ws === ws1

        # A different blocksize invalidates: Workspaces are sized by Nprof, and
        # reusing an undersized one would silently corrupt the chunk fill.
        hp2, _ = _plans!(cache, params, 128, 1)
        @test hp2 !== hp1
        @test cache.Nprof == 128

        # Different params, likewise.  Reuse is keyed on object identity, so an
        # equal-but-distinct SearchParams also rebuilds (conservative, correct).
        other = SearchParams(nharms = 8, metric = :sd2)
        hp3, _ = _plans!(cache, other, 128, 1)
        @test hp3 !== hp2
        @test cache.params === other

        # More tasks than before tops the vector up rather than rebuilding.
        _, ws4 = _plans!(cache, other, 128, 3)
        @test length(ws4) >= 3
        @test all(w isa Workspace for w in ws4)
        # …and the extra workspaces persist for a later, smaller request.
        _, ws5 = _plans!(cache, other, 128, 1)
        @test ws5 === ws4
    end
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
