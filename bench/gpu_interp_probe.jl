# What limits the GPU interpolation kernel?  (`docs/gpu_design.md` §4, stage 1)
#
#   julia --project=<env with CoherentSearch + CUDA> bench/gpu_interp_probe.jl
#
# Three arms with IDENTICAL arithmetic and identical access *shape*, differing
# only in where the interpolation weight comes from:
#
#   real      gW[o + (j-1)*V + lane]   per-lane   -- 128 B per warp per iteration
#   broadcast gW[o + (j-1)*V + 1]      same value -- 4 B per warp per iteration
#   constant  a register                          -- no weight load at all
#
# `broadcast` is numerically WRONG on purpose; it exists only to vary the traffic
# by 32x while holding everything else fixed.
#
# RESULT on a GTX 1080 (2026-08-25), ns per (harmonic, trial):
#
#   real 0.1773   broadcast 0.1766   constant 0.1027
#
# **real == broadcast, so weight-table VOLUME is irrelevant** -- which refutes the
# obvious "it is L2 bandwidth" reading of the kernel's 407 GB/s apparent traffic
# against a 237 GB/s device copy.  But removing the load *entirely* is 1.73x, so
# the cost is the load's issue slot and latency, not the bytes it moves.
#
# Two consequences, both recorded so they are not re-guessed:
#   - Staging the weight block in SHARED MEMORY would cut bandwidth we are not
#     short of.  It can only help through lower latency, not through volume.
#   - What would actually pay is removing the load from the inner loop: a warp
#     holding one group's weights in REGISTERS across the repeats that share its
#     group index (the residue cycles with period `ngrp`, so groups gi and
#     gi+ngrp use the identical block).  1.73x is the ceiling that buys.
#
# The real kernel measures 0.270 ns against this idealised loop's 0.177, so
# roughly a third of it is per-group setup and store, not the inner sum.

# Is the interpolation kernel limited by WEIGHT-TABLE traffic out of L2?
# Three arms, identical arithmetic, differing only in where the weight comes from:
#   real      : gW[o + (j-1)*V + lane]   -- per-lane, 128 B per warp per j
#   broadcast : gW[o + (j-1)*V + 1]      -- all lanes the same weight (WRONG result,
#                                           but ~32x less weight traffic)
#   constant  : a register               -- no weight traffic at all
# If "real" is much slower than "broadcast", the weight stream is the limit.
using CUDA, Printf

@inline function body(gW, amps, wi, ax, nj, V, mode)
    ar = 0.0f0; ai = 0.0f0
    @inbounds for _ in Int32(1):nj
        a = amps[ax]
        w = mode == 0 ? gW[wi] : (mode == 1 ? gW[wi - (wi % V)+ 1] : 1.0001f0)
        ar = fma(w, real(a), ar); ai = fma(w, imag(a), ai)
        wi += V; ax += Int64(1)
    end
    return ar, ai
end

function kern(out, gW, amps, ngroups::Int32, nharms::Int32, nj::Int32,
              ::Val{V}, ::Val{MODE}, nW::Int32, namps::Int64) where {V,MODE}
    lane = threadIdx().x
    gi = (blockIdx().x - Int32(1)) * blockDim().y + threadIdx().y
    h  = blockIdx().y
    gi <= ngroups || return nothing
    @inbounds begin
        # same access SHAPE as the real kernel: group's weight block cycles with
        # a small period, amplitudes advance with the trial index
        g = (gi - Int32(1)) % Int32(15)
        wi = (Int32(h - 1) * Int32(15) + g) * V * nj + lane
        wi = (wi % (nW - V*nj)) + 1
        ax = Int64(((gi - 1) * V) ÷ 2 + h) % (namps - Int64(nj)) + 1
        ar, ai = body(gW, amps, wi, ax, nj, V, MODE)
        jcol = (gi - Int32(1)) * V + lane
        out[jcol, h] = ar + ai
    end
    return nothing
end

const V = 32
nharms = 60; ngroups = 2048; nj = Int32(24)
nW = Int32(60 * 15 * V * 32)                      # ~1.4 MB, like the real table
gW = CUDA.rand(Float32, nW)
amps = CUDA.rand(ComplexF32, 1 << 22)
out = CUDA.zeros(Float32, ngroups * V, nharms)
best(f, n=7) = minimum(begin CUDA.@sync f(); t=time_ns(); CUDA.@sync f(); (time_ns()-t)/1e9 end for _ in 1:n)

println("weight-table traffic probe  (nj=$nj, V=$V, ngroups=$ngroups, nharms=$nharms)")
println("  arm          ms     ns/(harm,trial)   vs real")
const BASE = Ref(0.0)
for (name, mode) in (("real", 0), ("broadcast", 1), ("constant", 2))
    f() = @cuda threads=(V,8) blocks=(cld(ngroups,8), nharms) kern(
        out, gW, amps, Int32(ngroups), Int32(nharms), nj, Val(V), Val(mode),
        nW, Int64(length(amps)))
    t = best(f)
    ns = t*1e9/(ngroups*V*nharms)
    mode == 0 && (BASE[] = ns)
    @printf("  %-10s %7.3f   %8.4f        %5.2fx\n", name, t*1e3, ns, BASE[]/ns)
end
