# Micro-benchmark for the `:boxcar` gate — the zero-baseline width×phase scan
# that ~99% of trials return from, and the single largest bucket in the search.
#
# Sweeps the two axes that the production kernel (`_boxcar_gate!` in
# `src/search.jl`) commits to, so both stay re-measurable:
#
#   * the SIMD *batch* width `B` (profiles scanned at once, cross-profile),
#     including the `B` = runtime-value case that motivated making it a `Val`;
#   * the tile eltype, `Float32` (production) vs `Float64`.
#
# The reference is the per-column scalar path (`_boxcar_psum!` + `_boxcar_scan`),
# which vectorises along *phase* — fine at the base `nbins = 120`, useless at the
# `nbins = 20` the k=6 decimation folds to.
#
#     julia --project=bench -t 1 bench/boxcar_bench.jl
#
# Single-threaded by construction; run it that way for clean attribution.

using CoherentSearch
using BenchmarkTools
using Random
const CS = CoherentSearch

const NHARMS = 60
const MAXDECIM = 6
const NPROF = 2048          # the production chunk size
const BS = (16, 32, 48, 64, 96, 128)   # 64 is the production knee; see §3.1

# --- the scalar reference, per profile column -------------------------------
function gate_scalar!(out::Vector{Float64}, profs::Matrix{Float64}, n::Int,
                      psum::Vector{Float64}, widths::Vector{Int},
                      nbins::Int, invsigma::Float64)
    wmax = widths[end]
    @inbounds for j in 1:n
        col = @view profs[:, j]
        CS._boxcar_psum!(psum, col, nbins, wmax, 0.0)
        out[j] = CS._boxcar_scan(psum, widths, nbins, invsigma)
    end
    return out
end

# --- batched, parameterised on tile eltype and (static) batch width ---------
@inline function btranspose!(tile::Vector{T}, profs::Matrix{Float64},
                             j0::Int, nbins::Int, ::Val{B}) where {T,B}
    @inbounds for i in 1:nbins
        o = (i - 1) * B
        for b in 1:B
            tile[o + b] = T(profs[i, j0 + b])
        end
    end
end

@inline function bscan!(res::Vector{T}, mbuf::Vector{T}, tile::Vector{T},
                        psT::Vector{T}, widths::Vector{Int}, nbins::Int,
                        invsigma::T, ::Val{B}) where {T,B}
    wmax = widths[end]
    @inbounds for b in 1:B
        psT[b] = zero(T)
    end
    @inbounds for i in 1:(nbins + wmax)
        idx = i > nbins ? i - nbins : i
        o = (i - 1) * B
        t = (idx - 1) * B
        @simd for b in 1:B
            psT[o + B + b] = psT[o + b] + tile[t + b]
        end
    end
    @inbounds for b in 1:B
        res[b] = T(-Inf)
    end
    @inbounds for w in widths
        invsw = invsigma / sqrt(T(w))
        wo = w * B
        @simd for b in 1:B
            mbuf[b] = psT[wo + b] - psT[b]
        end
        for p in 2:nbins
            o = (p - 1) * B
            @simd for b in 1:B
                d = psT[o + wo + b] - psT[o + b]
                mbuf[b] = ifelse(d > mbuf[b], d, mbuf[b])
            end
        end
        @simd for b in 1:B
            c = mbuf[b] * invsw
            res[b] = ifelse(c > res[b], c, res[b])
        end
    end
end

function gate_batched!(out::Vector{Float64}, profs::Matrix{Float64}, n::Int,
                       tile::Vector{T}, psT::Vector{T}, res::Vector{T},
                       mbuf::Vector{T}, widths::Vector{Int}, nbins::Int,
                       invsigma::T, ::Val{B}) where {T,B}
    j0 = 0
    while j0 + B <= n
        btranspose!(tile, profs, j0, nbins, Val(B))
        bscan!(res, mbuf, tile, psT, widths, nbins, invsigma, Val(B))
        @inbounds for b in 1:B
            out[j0 + b] = Float64(res[b])
        end
        j0 += B
    end
    return out
end

# The same, with `B` passed as an ordinary runtime Int — the version that made
# batching *lose* to the scalar path until `B` was made a compile-time constant.
function gate_batched_dyn!(out::Vector{Float64}, profs::Matrix{Float64}, n::Int,
                           tile::Vector{T}, psT::Vector{T}, res::Vector{T},
                           mbuf::Vector{T}, widths::Vector{Int}, nbins::Int,
                           invsigma::T, B::Int) where {T}
    wmax = widths[end]
    j0 = 0
    while j0 + B <= n
        @inbounds for i in 1:nbins, b in 1:B
            tile[(i - 1) * B + b] = T(profs[i, j0 + b])
        end
        @inbounds for b in 1:B
            psT[b] = zero(T)
        end
        @inbounds for i in 1:(nbins + wmax)
            idx = i > nbins ? i - nbins : i
            o = (i - 1) * B; t = (idx - 1) * B
            @simd for b in 1:B
                psT[o + B + b] = psT[o + b] + tile[t + b]
            end
        end
        @inbounds for b in 1:B
            res[b] = T(-Inf)
        end
        @inbounds for w in widths
            invsw = invsigma / sqrt(T(w)); wo = w * B
            @simd for b in 1:B
                mbuf[b] = psT[wo + b] - psT[b]
            end
            for p in 2:nbins
                o = (p - 1) * B
                @simd for b in 1:B
                    d = psT[o + wo + b] - psT[o + b]
                    mbuf[b] = ifelse(d > mbuf[b], d, mbuf[b])
                end
            end
            @simd for b in 1:B
                c = mbuf[b] * invsw
                res[b] = ifelse(c > res[b], c, res[b])
            end
        end
        @inbounds for b in 1:B
            out[j0 + b] = Float64(res[b])
        end
        j0 += B
    end
    return out
end

# ---------------------------------------------------------------------------
Random.seed!(1234)
schedule = [2 * fld(NHARMS, k) for k in decimation_set(NHARMS, MAXDECIM)]
us(x) = rpad(string(round(x / 1e3, digits=1)), 9)

println("`:boxcar` gate, µs per $NPROF-profile chunk, single thread")
println("nharms=$NHARMS maxdecim=$MAXDECIM  ->  nbins schedule $schedule")
println("production kernel: tile=", CS._BC_TILE, "  B=", CS._BC_BATCH)
println("="^104)

for T in (Float64, Float32)
    println("\ntile eltype = $T")
    println(rpad("nbins", 7), rpad("nw", 4), rpad("scalar", 9),
            join(rpad("B=$b", 9) for b in BS), rpad("B=$(CS._BC_BATCH) dyn", 10), "best")
    tot_scalar = 0.0
    tot = zeros(length(BS))
    for nbins in schedule
        widths = boxcar_widths(nbins)
        wmax = widths[end]
        profs = randn(nbins, NPROF)
        profs .-= sum(profs; dims=1) ./ nbins        # DC = 0, as the real profiles are
        out1 = Vector{Float64}(undef, NPROF)
        out2 = Vector{Float64}(undef, NPROF)
        psum = Vector{Float64}(undef, nbins + wmax + 1)

        ts = minimum(@benchmark gate_scalar!($out1, $profs, $NPROF, $psum, $widths,
                                             $nbins, 1.0)).time
        gate_scalar!(out1, profs, NPROF, psum, widths, nbins, 1.0)

        row = Float64[]
        tdyn = 0.0
        for B in BS
            tile = Vector{T}(undef, B * nbins)
            psT = Vector{T}(undef, B * (nbins + wmax + 1))
            res = Vector{T}(undef, B); mbuf = Vector{T}(undef, B)
            vb = Val(B)
            push!(row, minimum(@benchmark gate_batched!($out2, $profs, $NPROF, $tile, $psT,
                                                        $res, $mbuf, $widths, $nbins,
                                                        $(one(T)), $vb)).time)
            gate_batched!(out2, profs, NPROF, tile, psT, res, mbuf, widths, nbins, one(T), vb)
            nfull = (NPROF ÷ B) * B
            if T === Float64      # F64 batching must be bit-identical to the scalar path
                @assert out2[1:nfull] == out1[1:nfull] "B=$B not bit-identical at nbins=$nbins"
            end
            if B == CS._BC_BATCH
                tdyn = minimum(@benchmark gate_batched_dyn!($out2, $profs, $NPROF, $tile,
                                                            $psT, $res, $mbuf, $widths,
                                                            $nbins, $(one(T)), CS._BC_BATCH)).time
            end
        end
        println(rpad(nbins, 7), rpad(length(widths), 4), us(ts), join(us(t) for t in row),
                rpad(round(tdyn / 1e3, digits=1), 10), round(ts / minimum(row), digits=2), "x")
        tot_scalar += ts
        tot .+= row
    end
    println("-"^104)
    println(rpad("TOTAL", 11), us(tot_scalar), join(us(t) for t in tot),
            rpad("", 10), round(tot_scalar / minimum(tot), digits=2), "x")
end

# --- how far the Float32 gate can stray from the exact Float64 bound --------
println("\n", "="^104)
println("Float32 gate error vs the Float64 gate (this is what must stay under boxcar_medmargin)")
for nbins in schedule
    widths = boxcar_widths(nbins)
    wmax = widths[end]
    profs = randn(nbins, NPROF)
    for j in 1:8:NPROF, i in 1:max(1, nbins ÷ 20)      # some profiles carry a pulse
        profs[i, j] += 5.0
    end
    profs .-= sum(profs; dims=1) ./ nbins
    o64 = Vector{Float64}(undef, NPROF); o32 = Vector{Float64}(undef, NPROF)
    gate_scalar!(o64, profs, NPROF, Vector{Float64}(undef, nbins + wmax + 1), widths, nbins, 1.0)
    B = CS._BC_BATCH
    gate_batched!(o32, profs, NPROF, Vector{Float32}(undef, B * nbins),
                  Vector{Float32}(undef, B * (nbins + wmax + 1)),
                  Vector{Float32}(undef, B), Vector{Float32}(undef, B),
                  widths, nbins, 1.0f0, Val(B))
    nfull = (NPROF ÷ B) * B
    println(rpad("nbins=$nbins", 12), "max |Δ| = ",
            rpad(maximum(abs.(o32[1:nfull] .- o64[1:nfull])), 24),
            "  margin/err = ",
            round(2.0 / maximum(abs.(o32[1:nfull] .- o64[1:nfull])); sigdigits=3))
end
