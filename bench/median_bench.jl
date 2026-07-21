# Compare exact-median strategies on COLD, varied small columns — the real
# access pattern of _profile_snr (a different profile every call), not the
# same-column hot loop that flatters the radix sort.
#
#     julia --project=bench -t 1 bench/median_bench.jl [FILE.fft]

using CoherentSearch
using BenchmarkTools
using Statistics: median!
const CS = CoherentSearch

const FILE = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"

ft = FFTFile(FILE)
nharms = 60
params = SearchParams(nharms=nharms, threshold=6.0, metric=:boxcar,
                      decimations=decimation_set(nharms, 6))
Nprof = 2048
lodr  = params.hidr / params.nharms
rstart = 10.0 * ft.T
hplans = CS.build_harmonic_plans(params, Nprof)
ws     = CS.Workspace(params, hplans, Nprof)
CS.fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, Nprof)  # real profiles

# --- median strategies (all return the mean of the two central order stats) ---

# (0) current: copy then full radix sort
@inline function med_sort!(buf, col, nbins)
    copyto!(buf, col)
    sort!(buf; alg=QuickSort)
    half = nbins >>> 1
    return 0.5 * (buf[half] + buf[half+1])
end

# (1) hand-written Hoare quickselect, median-of-3 pivot; selects the upper
#     median in place, lower median = max of the left partition.
@inline _swap!(v, a, b) = (@inbounds t = v[a]; @inbounds v[a] = v[b]; @inbounds v[b] = t; nothing)
# Quickselect (Lomuto, median-of-3 pivot, insertion-sort cutoff): after the
# call, v[k] holds the k-th smallest of v[lo:hi] and v[lo:k-1] are all ≤ it.
@inline function _sel!(v, lo, hi, k)
    @inbounds while lo < hi
        if hi - lo < 16                      # small range: just sort it
            for i in lo+1:hi
                x = v[i]; j = i - 1
                while j >= lo && v[j] > x
                    v[j+1] = v[j]; j -= 1
                end
                v[j+1] = x
            end
            return
        end
        mid = (lo + hi) >>> 1                 # median-of-3 -> pivot at v[hi]
        v[mid] < v[lo] && _swap!(v, mid, lo)
        v[hi]  < v[lo] && _swap!(v, hi, lo)
        v[mid] < v[hi] && _swap!(v, mid, hi)
        pivot = v[hi]
        i = lo - 1
        for j in lo:hi-1
            if v[j] <= pivot
                i += 1; _swap!(v, i, j)
            end
        end
        _swap!(v, i+1, hi)
        p = i + 1
        p == k && return
        p < k ? (lo = p + 1) : (hi = p - 1)
    end
end
@inline function med_qselect!(buf, col, nbins)
    copyto!(buf, col)
    half = nbins >>> 1
    _sel!(buf, 1, nbins, half+1)               # upper median at buf[half+1]
    upper = buf[half+1]
    lower = -Inf                               # lower median = max of buf[1:half]
    @inbounds for i in 1:half
        buf[i] > lower && (lower = buf[i])
    end
    return 0.5 * (lower + upper)
end

# (2) Statistics.median!
@inline function med_stats!(buf, col, nbins)
    copyto!(buf, col)
    return median!(buf)
end

# (3) insertion sort (baseline; expected poor for n=120)
@inline function med_insertion!(buf, col, nbins)
    copyto!(buf, col)
    @inbounds for i in 2:nbins
        x = buf[i]; j = i - 1
        while j >= 1 && buf[j] > x
            buf[j+1] = buf[j]; j -= 1
        end
        buf[j+1] = x
    end
    half = nbins >>> 1
    return 0.5 * (buf[half] + buf[half+1])
end

function run_all(f, profs, buf, nbins, N)
    s = 0.0
    @inbounds for j in 1:N
        s += f(buf, view(profs, :, j), nbins)
    end
    s
end

profs = ws.profs
nbins = 2nharms
buf = Vector{Float64}(undef, nbins)

# correctness: all must match the sort reference exactly
ref = run_all(med_sort!, profs, buf, nbins, Nprof)
for (nm, f) in (("qselect", med_qselect!), ("stats", med_stats!), ("insertion", med_insertion!))
    got = run_all(f, profs, buf, nbins, Nprof)
    @assert got ≈ ref "median mismatch: $nm  ($got vs $ref)"
end
println("all median strategies agree with sort reference ✓  (nbins=$nbins, cold columns)\n")

for (nm, f) in (("sort! (current)", med_sort!), ("quickselect", med_qselect!),
                ("Statistics.median!", med_stats!), ("insertion", med_insertion!))
    b = @benchmark run_all($f, $profs, $buf, $nbins, $Nprof)
    println(rpad(nm, 22), ": ", BenchmarkTools.prettytime(minimum(b).time),
            "  => ", round(minimum(b).time/Nprof; digits=1), " ns/median")
end
