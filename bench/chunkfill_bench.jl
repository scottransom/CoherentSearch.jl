# Split `fill_chunk_profiles!` into its three phases and time each at both
# profile precisions, on the real `Workspace`/`DirectPlan`s.
#
#     julia --project=bench -t 1 bench/chunkfill_bench.jl [FILE.fft] [freq_Hz]
#
# This exists because the coarse phase timers in `search.jl` localise a cost to
# "interp" or "brfft" but cannot say whether it is the zeroing, the O(m) inner
# sum, or the strided scatter into `ftprofs`.  Each phase here is measured on the
# same buffers, in the same order, that the search uses — the arrays are the
# production ones, so cache state is representative even though the surrounding
# metric work is absent.

using CoherentSearch
using BenchmarkTools
using LinearAlgebra: mul!
using Printf
const CS = CoherentSearch

const FILE  = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"
const FREQ  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
const NHARMS = 60
const NPROF  = 2048

ft = FFTFile(FILE)
rstart = FREQ * ft.T

const MAXDECIM = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6

mkparams(prec) = SearchParams(nharms=NHARMS, threshold=6.0,
                              decimations=decimation_set(NHARMS, MAXDECIM),
                              precision=prec)

struct Arm
    prec::Symbol
    params::SearchParams
    ws::Any
    dplans::Vector{<:DirectPlan}
end

function arm(prec)
    params = mkparams(prec)
    ws = CS.Workspace(params, NPROF)
    dplans = CS.build_direct_plans(params, rstart)
    Arm(prec, params, ws, dplans)
end

# The three phases, callable in isolation.
zero_only!(a) = fill!(a.ws.ftprofs, 0)
function interp_only!(a)
    for dp in a.dplans
        CS.fill_harmonic_row_direct!(a.ws, dp, ft, a.params, 0, NPROF)
    end
end
brfft_only!(a) = mul!(a.ws.profs, a.ws.brfftplan, a.ws.ftprofs)

us(b) = minimum(b).time / 1000

function main()
    arms = [arm(:f64), arm(:f32)]
    @printf("chunk fill — %s  f=%.3g Hz  nharms=%d  Nprof=%d  m=%d\n",
            FILE, FREQ, NHARMS, NPROF, arms[1].params.m)
    @printf("  ftprofs %d×%d: %s = %.0f KB   |   %s = %.0f KB\n\n",
            NHARMS + 1, NPROF,
            eltype(arms[1].ws.ftprofs), sizeof(arms[1].ws.ftprofs) / 1024,
            eltype(arms[2].ws.ftprofs), sizeof(arms[2].ws.ftprofs) / 1024)

    phases = ["fill!(ftprofs,0)" => zero_only!,
              "interp (60 rows)" => interp_only!,
              "brfft"            => brfft_only!,
              "all three"        => a -> (zero_only!(a); interp_only!(a); brfft_only!(a))]

    @printf("  %-20s %10s %10s %10s\n", "phase", "f64 µs", "f32 µs", "f32/f64")
    for (name, f) in phases
        t = [us(@benchmark $f($a) evals=1 samples=200) for a in arms]
        @printf("  %-20s %10.1f %10.1f %9.2fx\n", name, t[1], t[2], t[2] / t[1])
    end

    # --- The decimated transforms, on genuine chunk contents -----------------
    # `ftprofs` now holds a real chunk (the "all three" phase above left it
    # there), and `db.src` is a stride-k view of it — the decimated stack, with
    # no copy.  A random-filled array of the same shape is timed alongside: FFTW
    # is data-independent apart from subnormals, which real amplitudes never
    # reach, so these are shape-and-stride measurements.
    isempty(arms[1].ws.decims) && return
    println()
    @printf("  %-20s %10s %10s %10s\n", "decim phase", "f64 \u00b5s", "f32 \u00b5s", "f32/f64")
    for di in eachindex(arms[1].ws.decims)
        ks = arms[1].ws.decims[di].k
        tb = Float64[]
        for a in arms
            db = a.ws.decims[di]
            push!(tb, us(@benchmark mul!($(db.dprofs), $(db.brfftplan), $(db.src)) evals=1 samples=200))
        end
        @printf("  k=%-2d brfft (strided)%3s %10.1f %10.1f %9.2fx\n", ks, "", tb[1], tb[2], tb[2]/tb[1])
    end
end

main()
