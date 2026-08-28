# Standalone GPU classifier for the CoherentSearch.jl GPU track, in the spirit of
# `bench/avx512_probe.jl`: no package, no data, no root.  Send it to any GPU host
# that needs classifying.
#
#   julia --project=<env-with-CUDA> bench/gpu_probe.jl
#
# It reports, in order:
#   1. device identity and what fraction of FP32 peak and of memory bandwidth is
#      actually reachable on this card (so every later number has a denominator);
#   2. cuFFT batched C2R throughput at the *search's* transform sizes — the base
#      120-bin fold and the five decimated rungs — in ns per output bin, which is
#      directly comparable to the `bench/decim_brfft_bench.jl` figure for FFTW;
#   3. a hand-rolled fused direct-DFT alternative to cuFFT for the k=1 fold,
#      measured and accuracy-checked against cuFFT.
#
# (3) exists because the staged GPU pipeline is transform-dominated (see
# `docs/gpu_design.md` §2), so "can we beat cuFFT at n=120?" is the load-bearing
# question for the whole design.  On a GTX 1080 the answer is a clear NO — the
# DFT is ~3.5x slower despite using only 5% of the FP32 peak, i.e. it is
# shared-memory/latency bound, not FLOP bound.  Re-run it on any new card before
# assuming that verdict carries.
#
# TRAP: create cuFFT plans OUTSIDE the timing closure.  Planning inside it costs
# more than the transform and produced a confidently wrong 7x once already.

using CUDA, CUDA.CUFFT, Printf, LinearAlgebra

bestof(f, n = 7) = minimum(begin
    CUDA.synchronize(); t0 = time_ns(); f(); CUDA.synchronize(); (time_ns() - t0) / 1e9
end for _ in 1:n)

const NB = 120        # base fold: 2*nharms bins
const NH = 61         # nharms + 1 stored harmonics (DC held at zero)
const TPB = 8         # trials per block in the DFT kernel
const NACC = 4        # independent accumulators, for FMA-chain ILP

# --------------------------------------------------------------------------
# 1. What this card can actually do
# --------------------------------------------------------------------------
# Top-level, NOT nested inside `capabilities()`: an inner function that shares a
# name with an assigned local of its enclosing scope makes Julia box the variable,
# and a boxed capture is not a bitstype, so the kernel will not compile at all.
# Same trap `src/directinterp.jl` records for `_group_lanes` (there it was 2000x
# slower rather than a hard error).
function _fmakern(out, niter)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    bb = 1.0000001f0; cc = 0.9999999f0
    d = Float32(i); e = d * 0.5f0; f = d * 0.25f0; g = d * 0.125f0
    for _ in 1:niter
        d = fma(d, bb, cc); e = fma(e, bb, cc); f = fma(f, bb, cc); g = fma(g, bb, cc)
    end
    @inbounds out[i] = d + e + f + g
    return
end

# FP32 CUDA cores per SM, by compute capability.  Needed to turn the measured
# GFLOP/s into a "% of peak", which is the only form comparable across cards.
function cores_per_sm(cap)
    v = (cap.major, cap.minor)
    v == (6, 0) && return 64            # Pascal GP100
    v[1] == 6 && return 128             # Pascal GP10x
    v[1] == 7 && return 64              # Volta / Turing
    v == (8, 0) && return 64            # Ampere GA100
    v[1] == 8 && return 128             # Ampere GA10x / Ada
    v[1] == 9 && return 128             # Hopper
    v[1] >= 10 && return 128            # Blackwell and later (check if it matters)
    return 0                            # unknown -> peak reported as n/a
end

function capabilities()
    dev = device()
    cap = CUDA.capability(dev)
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    clk = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) / 1e6      # GHz
    buswidth = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    memclk = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) / 1e6  # GHz
    cps = cores_per_sm(cap)
    peak_gflops = cps == 0 ? NaN : 2 * cps * sms * clk
    peak_gbs = 2 * memclk * buswidth / 8         # DDR: 2 transfers per clock
    l2 = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    println("device: ", name(dev), "  sm_", cap, "  SMs=", sms,
            @sprintf("  clock=%.3f GHz  cores/SM=%d", clk, cps))
    @printf("L2    : %.1f MB   <- the knee in the cuFFT sweep below should sit here\n",
            l2 / 2^20)
    free, tot = CUDA.memory_info()
    @printf("memory: %.2f GiB free of %.2f GiB   bus %d-bit @ %.3f GHz\n",
            free / 2^30, tot / 2^30, buswidth, memclk)
    @printf("peak  : FP32 %.0f GFLOP/s   bandwidth %.0f GB/s\n", peak_gflops, peak_gbs)

    nth, nbl, nit = 256, sms * 32, 4096
    out = CUDA.zeros(Float32, nth * nbl)
    t = bestof(() -> @cuda(threads = nth, blocks = nbl, _fmakern(out, nit)))
    gflops = 2.0 * 4 * nit * nth * nbl / t / 1e9
    @printf("FP32 FMA kernel : %7.3f ms -> %6.0f GFLOP/s  (%.0f%% of peak)\n",
            t * 1e3, gflops, 100 * gflops / peak_gflops)

    n = 1 << 26
    a = CUDA.rand(Float32, n); b = CUDA.zeros(Float32, n)
    t = bestof(() -> copyto!(b, a))
    gbs = 2 * 4n / t / 1e9
    @printf("device copy     : %7.3f ms -> %6.0f GB/s     (%.0f%% of peak)\n",
            t * 1e3, gbs, 100 * gbs / peak_gbs)
    CUDA.unsafe_free!(out); CUDA.unsafe_free!(a); CUDA.unsafe_free!(b)
    return gflops, gbs, l2
end

# --------------------------------------------------------------------------
# 2. cuFFT at the search's sizes, swept over `Nprof`.
#
# The FFTW reference is 2.048 ns per output bin for the contiguous base pass on
# one Xeon Silver 4114 core (:f64), from `bench/decim_brfft_bench.jl`.
#
# **The "eff GB/s" column is measured against the DRAM copy above, so a value
# over 100% is not an error -- it means the transform's working set fits in L2
# and never touched DRAM.**  That is the single most useful thing this table
# says, because it locates the `Nprof` at which the whole transform stage stays
# in cache, and on a large-L2 card (Ada, Hopper) that is worth more than any
# kernel tuning.  It also inverts the GTX 1080's guidance, where bigger `Nprof`
# was uniformly better -- so **do not carry an `Nprof` choice between cards.**
#
# The last column is the decision-relevant one: the whole six-rung transform
# stage for the reference workload (PM0063 at the riptide bench config,
# 8,363,442 trial fundamentals x 294 output bins each), derived from that row's
# per-bin costs.  Compare it against the CPU's ~0.51 s at -t 20.
# --------------------------------------------------------------------------
const FFTW_NS_PER_BIN = 2.048
const BENCH_TRIALS = 8_363_442          # PM0063, 0.1-33.333 Hz, hidr/nharms grid
const LADDER = ((1, 60), (2, 30), (3, 20), (4, 15), (5, 12), (6, 10))

function cufft_ladder(achievable_gbs, l2;
                      Nprofs = (16384, 32768, 65536, 131072, 262144))
    println("\ncuFFT batched C2R, swept over Nprof   (L2 = ",
            @sprintf("%.1f MB", l2 / 2^20), ")")
    println("  Nprof    k  nbins   workset       ms   ns/out-bin   vs FFTW   eff GB/s   %DRAM")
    best_t, best_n = Inf, 0
    for Nprof in Nprofs
        stage_ns = 0.0
        for (k, Hk) in LADDER
            nb = 2Hk
            src = CUDA.zeros(ComplexF32, Hk + 1, Nprof)
            dst = CUDA.zeros(Float32, nb, Nprof)
            plan = plan_brfft(src, nb, 1)      # OUTSIDE the timing closure
            t = bestof(() -> mul!(dst, plan, src))
            ns = t * 1e9 / (nb * Nprof)
            stage_ns += nb * ns                # ns per trial fundamental, this rung
            bytes = Nprof * ((Hk + 1) * 8 + nb * 4)
            gbs = bytes / t / 1e9
            @printf("  %7d  %d  %5d  %6.1f MB  %7.3f    %8.5f  %6.1fx     %6.0f    %4.0f%%%s\n",
                    Nprof, k, nb, bytes / 2^20, t * 1e3, ns, FFTW_NS_PER_BIN / ns,
                    gbs, 100 * gbs / achievable_gbs,
                    gbs > achievable_gbs ? "  <- L2-resident" : "")
            CUDA.unsafe_free!(src); CUDA.unsafe_free!(dst)
        end
        stage_s = stage_ns * BENCH_TRIALS / 1e9
        stage_s < best_t && ((best_t, best_n) = (stage_s, Nprof))
        @printf("  %7d  ---> whole six-rung transform stage for the reference workload: %.3f s\n\n",
                Nprof, stage_s)
    end
    @printf("BEST Nprof for the transform stage: %d at %.3f s", best_n, best_t)
    @printf("  (CPU -t 20 spends ~0.51 s here, so %.1fx)\n", 0.51 / best_t)
    return best_n
end

# --------------------------------------------------------------------------
# 3. The fused direct-DFT alternative, for the k=1 fold.
#
# One block = NB threads, thread p owning output phase p, so A[t,h] broadcasts to
# every thread and only the twiddle lookup differs per thread.  `idx = (h*p) mod
# NB` is carried incrementally (exact integer, no drift).  Harmonics are split
# into NACC contiguous blocks with independent accumulators: a single accumulator
# makes all 2*(NH-2) FMAs one dependent chain.
#
# The c2r convention matched here is FFTW's unnormalised `brfft`:
#   out[p] = X0 + 2*sum_{h=1}^{NB/2-1} (Ar_h cos - Ai_h sin) + X_nyq*(-1)^p
# --------------------------------------------------------------------------
function dftkern(dst, src, Nprof)
    tw = CuDynamicSharedArray(Float32, 2 * NB)
    A  = CuDynamicSharedArray(Float32, 2 * NH * TPB, 2 * NB * sizeof(Float32))
    p = threadIdx().x - 1
    @inbounds begin
        ang = 2.0f0 * Float32(pi) * Float32(p) / Float32(NB)
        tw[p + 1] = cos(ang); tw[NB + p + 1] = sin(ang)
    end
    t0 = (blockIdx().x - 1) * TPB
    sync_threads()
    srcf = reinterpret(Float32, src)
    @inbounds begin
        i = p
        while i < 2 * NH * TPB                       # coalesced tile load
            col = i ÷ (2 * NH); off = i % (2 * NH); g = t0 + col
            A[i + 1] = g < Nprof ? srcf[g * 2 * NH + off + 1] : 0.0f0
            i += NB
        end
        sync_threads()
        hlo = (1, 16, 31, 46); hhi = (15, 30, 45, 59)
        for c in 0:(TPB - 1)
            g = t0 + c
            g < Nprof || continue
            base = c * 2 * NH
            a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0
            for s in 1:NACC
                acc = 0.0f0
                idx = (hlo[s] * p) % NB
                for h in hlo[s]:hhi[s]
                    ar = A[base + 2h + 1]; ai = A[base + 2h + 2]
                    acc = fma(ar, tw[idx + 1], acc)
                    acc = fma(-ai, tw[NB + idx + 1], acc)
                    idx += p; idx >= NB && (idx -= NB)
                end
                s == 1 ? (a1 = acc) : s == 2 ? (a2 = acc) : s == 3 ? (a3 = acc) : (a4 = acc)
            end
            nyq = A[base + 2 * 60 + 1] * (iseven(p) ? 1.0f0 : -1.0f0)
            dst[p + 1 + g * NB] = A[base + 1] + 2.0f0 * (a1 + a2 + a3 + a4) + nyq
        end
    end
    return
end

function dft_vs_cufft(Nprofs = (65536, 262144))
    println("\nfused direct DFT vs cuFFT, k=1 fold (nbins=$NB, nharms+1=$NH)")
    println("  Nprof      DFT ms  ns/bin    cuFFT ms  ns/bin   cuFFT/DFT   rel err")
    for Nprof in Nprofs
        src = CuArray(ComplexF32.(randn(Float32, NH, Nprof), randn(Float32, NH, Nprof)))
        CUDA.@allowscalar src[1, :] .= 0            # DC held at zero, as the search does
        dst = CUDA.zeros(Float32, NB, Nprof)
        ref = CUDA.zeros(Float32, NB, Nprof)
        sh = (2 * NB + 2 * NH * TPB) * sizeof(Float32)
        run() = @cuda threads = NB blocks = cld(Nprof, TPB) shmem = sh dftkern(dst, src, Nprof)
        run(); CUDA.synchronize()
        plan = plan_brfft(src, NB, 1)               # OUTSIDE the timing closure
        td = bestof(run)
        tf = bestof(() -> mul!(ref, plan, src))
        err = maximum(abs.(Array(dst) .- Array(ref))) / maximum(abs.(Array(ref)))
        @printf("  %7d  %8.3f  %6.4f    %8.3f  %6.4f     %6.2fx   %.1e\n",
                Nprof, td * 1e3, td * 1e9 / (NB * Nprof),
                tf * 1e3, tf * 1e9 / (NB * Nprof), td / tf, err)
        CUDA.unsafe_free!(src); CUDA.unsafe_free!(dst); CUDA.unsafe_free!(ref)
    end
    println("  (cuFFT/DFT < 1 means cuFFT wins.  rel err is against cuFFT, and ~3e-7 is")
    println("   Float32 machine precision -- the DFT is correct, just not competitive.)")
end

CUDA.functional() || error("no functional CUDA device")
CUDA.versioninfo()
const GFLOPS, GBS, L2 = capabilities()
cufft_ladder(GBS, L2)
dft_vs_cufft()

println("\n" * "="^78)
println("Paste this line back with the card's name -- it is the whole classification:")
@printf("  %s | sm_%s | %.0f GFLOP/s | %.0f GB/s | L2 %.0f MB\n",
        name(device()), string(CUDA.capability(device())), GFLOPS, GBS, L2 / 2^20)
println("="^78)
