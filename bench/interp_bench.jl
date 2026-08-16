# Throughput comparison of the Fourier-interpolation methods.
#
#     julia --project=bench -t 1 bench/interp_bench.jl [FILE.fft]
#
# The question this answers: an FFT-correlation interpolator is O(K log K) for K
# grid points while direct summation is O(N m) for N points, so surely the FFT
# wins whenever a lot of points are wanted on a uniform grid?  The answer is
# "only in a corner of the parameter space we do not occupy", for two reasons
# this benchmark separates cleanly:
#
#   1. **Per-point cost.**  With a real-weighted kernel (see `directinterp.jl`)
#      one direct point is `m` real FMAs = 4m flops.  One FFT *grid* point costs
#      two transforms plus a multiply, ~`10 log2(fftlen) + 6` flops.  These cross
#      at `m ≈ 2.5 log2(fftlen)` — about m = 35 at fftlen = 16384.  At the
#      production m = 32 they are already comparable, before any of point 2.
#
#   2. **Grid waste.**  The FFT can only produce points on a uniform grid of
#      `numbetween` points per Fourier bin, and that grid has to be fine enough
#      for the *linear interpolation* that follows to be accurate — not merely
#      fine enough to contain the points wanted.  In the search's own schedule
#      that forces `numbetween = 16` while the trials are `0.5` bins apart, so 8
#      grid points are computed for every point actually read.  Direct summation
#      has no grid and computes exactly what is asked for.
#
# `oversampling` below is that ratio (grid points per requested point).  The
# `grid` column reports the FFT's throughput counted over *all* the points it
# produces, which is the fair way to see that FFTW is not being cheated: it is
# fast per grid point, it is just making points nobody wants.
#
# Methods compared, all producing the same N points at the same frequencies:
#   fourier_interp  one `fourier_interp` call per point (the exact reference)
#   direct          `finterp_direct!` — exact, real-weight form, no phase table
#   direct_table    the search's hot loop (`fill_harmonic_row_direct!`), which
#                   also tabulates the weights over the search's finitely many dr
#   fft+linear      `finterp_fft` onto the fine grid, then `uniform_linear_interp`
#                   — the Python original's method and the `--interp fft` path
#
# Writes `bench/interp_bench_throughput.png` and `bench/interp_bench_crossover.png`.

using CoherentSearch
using BenchmarkTools
using Printf
using CairoMakie
const CS = CoherentSearch

const FILE = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"

amps = if isfile(FILE)
    FFTFile(FILE).amps
else
    @info "No .fft file given/found; using synthetic white amplitudes" FILE
    ComplexF32.(randn(ComplexF64, 1 << 21))
end
@info "Interpolation benchmark" file=(isfile(FILE) ? FILE : "synthetic") nbins=length(amps)

const R0 = 100_000.3141592          # a generic, non-special starting frequency

# --- the four methods -------------------------------------------------------

bench_reference(out, n, step, m) = begin
    @inbounds for k in 1:n
        out[k] = CS.fourier_interp(R0 + (k - 1) * step, amps, m)
    end
    out
end

bench_direct(out, n, step, m) = CS.finterp_direct!(out, R0, n, step, amps, m)

# The FFT arm is now measured through `finterp_fft`, the *reference* kernel that
# `reference_profiles` uses and the Python oracle is pinned to.  It allocates its
# grid, where the retired production `interp_tile!` reused prebuilt FFTW buffers,
# so this arm is a little pessimistic — which does not matter for its only
# remaining purpose, recording the order-of-magnitude gap that motivated the
# direct interpolator in the first place.  Do not quote it as a production
# figure; the settled number is the 3.8x in Summary_and_Future_Work.md.
"""FFT-correlation onto a `nb`-per-bin grid, then linear interpolation."""
function bench_fft_linear(out, n, step, m, nb, lobin, numbins, fftlen)
    grid = finterp_fft(lobin, numbins, nb, amps, m; fftlen=fftlen)
    @inbounds for k in 1:n
        out[k] = CS.uniform_linear_interp(R0 + (k - 1) * step, lobin, nb, grid)
    end
    return out
end

"""Cost of the FFT-correlation alone, i.e. per *grid* point."""
bench_fft_grid(m, nb, lobin, numbins, fftlen) =
    finterp_fft(lobin, numbins, nb, amps, m; fftlen=fftlen)

function fft_setup(n, step, m, nb)
    lobin = floor(Int, R0)
    numbins = floor(Int, R0 + (n - 1) * step) - lobin + 2
    fftlen = next_smooth((numbins + m) * nb)
    return (lobin, numbins, fftlen, numbins * nb)
end

# The search's own tabulated hot loop, driven through a real Workspace.
function direct_table_setup(n, step, m, nharms)
    params = SearchParams(nharms=nharms, m=m)
    ws = CS.Workspace(params, n)
    dplans = CS.build_direct_plans(params, R0 / nharms)
    return (params, ws, dplans[nharms])       # top harmonic: step = hidr = 0.5
end

# --- sweep ------------------------------------------------------------------

struct Row
    m::Int
    nb::Int
    n::Int
    over::Float64
    t_ref::Float64
    t_dir::Float64
    t_tab::Float64
    t_fft::Float64
    t_grid::Float64
    ngrid::Int
    err_dir::Float64
    err_fft::Float64
end

const STEP = 0.5     # bins per requested point (the search's top-harmonic step)

function run_row(m, nb, n; want_table::Bool)
    out = Vector{ComplexF64}(undef, n)
    ref = Vector{ComplexF64}(undef, n)
    lobin, numbins, fftlen, ngrid = fft_setup(n, STEP, m, nb)

    bench_reference(ref, n, STEP, m)
    t_ref = @belapsed bench_reference($out, $n, $STEP, $m) samples=12 evals=1
    t_dir = @belapsed bench_direct($out, $n, $STEP, $m) samples=12 evals=1
    bench_direct(out, n, STEP, m)
    err_dir = maximum(abs.(out .- ref) ./ abs.(ref))
    t_fft = @belapsed bench_fft_linear($out, $n, $STEP, $m, $nb, $lobin, $numbins, $fftlen) samples=12 evals=1
    bench_fft_linear(out, n, STEP, m, nb, lobin, numbins, fftlen)
    err_fft = maximum(abs.(out .- ref) ./ abs.(ref))
    t_grid = @belapsed bench_fft_grid($m, $nb, $lobin, $numbins, $fftlen) samples=12 evals=1

    t_tab = NaN
    if want_table
        # nharms = 60 fixes the trial step at hidr = 0.5 for the top harmonic.
        params, ws, dp = direct_table_setup(n, STEP, m, 60)
        ftf = _wrap_amps(amps)
        t_tab = @belapsed CS.fill_harmonic_row_direct!($ws, $dp, $ftf, $params, 0, $n) samples=12 evals=1
    end
    return Row(m, nb, n, nb * STEP, t_ref, t_dir, t_tab, t_fft, t_grid, ngrid, err_dir, err_fft)
end

# A minimal in-memory FFTFile so the Workspace path can run on `amps`.
function _wrap_amps(a)
    N = 2 * length(a)
    dt = 1.0e-4
    inf = SimpleInf("", nothing, nothing, N, dt, nothing)
    return FFTFile("", a, inf, N, N * dt, 1.0 / (N * dt), false, false,
                   real(a[1]), imag(a[1]))
end

mpps(n, t) = n / t / 1e6      # million points per second

println("\n", "="^108)
println("Throughput in million interpolated points/sec.  step = $STEP bins between requested points.")
println("oversampling = fine-grid points computed per requested point (nb*step).")
println("="^108)
@printf("%4s %4s %6s %6s | %9s %9s %9s %9s | %9s | %9s %9s\n",
        "m", "nb", "N", "over", "ref", "direct", "dir+tab", "fft+lin", "fft/grid",
        "err_dir", "err_fft")
println("-"^108)

rows = Row[]
for m in (8, 16, 32, 64, 128), nb in (2, 4, 8, 16, 32)
    r = run_row(m, nb, 2048; want_table = true)
    push!(rows, r)
    @printf("%4d %4d %6d %6.1f | %9.1f %9.1f %9s %9.1f | %9.1f | %9.1e %9.1e\n",
            r.m, r.nb, r.n, r.over,
            mpps(r.n, r.t_ref), mpps(r.n, r.t_dir),
            isnan(r.t_tab) ? "-" : @sprintf("%.1f", mpps(r.n, r.t_tab)),
            mpps(r.n, r.t_fft), mpps(r.ngrid, r.t_grid),
            r.err_dir, r.err_fft)
end

println("\n", "="^108)
println("N sweep at m = 32 (the production kernel width)")
println("="^108)
@printf("%4s %4s %6s %6s | %9s %9s %9s %9s | %9s\n",
        "m", "nb", "N", "over", "ref", "direct", "dir+tab", "fft+lin", "fft/grid")
println("-"^108)
nrows = Row[]
for n in (128, 512, 2048, 8192, 32768), nb in (2, 16)
    r = run_row(32, nb, n; want_table=true)
    push!(nrows, r)
    @printf("%4d %4d %6d %6.1f | %9.1f %9.1f %9.1f %9.1f | %9.1f\n",
            r.m, r.nb, r.n, r.over,
            mpps(r.n, r.t_ref), mpps(r.n, r.t_dir), mpps(r.n, r.t_tab),
            mpps(r.n, r.t_fft), mpps(r.ngrid, r.t_grid))
end

# --- save the raw rows so the plots can be redrawn without re-timing ---------
open(joinpath(@__DIR__, "interp_bench.csv"), "w") do io
    println(io, "set,m,nb,N,oversampling,t_ref,t_direct,t_table,t_fftlin,t_fftgrid,ngrid,err_direct,err_fft")
    for (name, rs) in (("m_sweep", rows), ("N_sweep", nrows)), r in rs
        @printf(io, "%s,%d,%d,%d,%g,%g,%g,%g,%g,%g,%d,%g,%g\n",
                name, r.m, r.nb, r.n, r.over, r.t_ref, r.t_dir, r.t_tab,
                r.t_fft, r.t_grid, r.ngrid, r.err_dir, r.err_fft)
    end
end

# --- interpretation ---------------------------------------------------------

println("\n", "="^108)
println("Where does the FFT actually win?")
println("="^108)
for m in (8, 16, 32, 64, 128)
    dense = only(filter(r -> r.m == m && r.nb == 2, rows))     # oversampling 1
    prod  = only(filter(r -> r.m == m && r.nb == 16, rows))    # oversampling 8
    @printf("m=%3d  dense grid (over=1): FFT per grid point %6.1f vs direct+table %6.1f  -> FFT %s\n",
            m, mpps(dense.ngrid, dense.t_grid), mpps(dense.n, dense.t_tab),
            mpps(dense.ngrid, dense.t_grid) > mpps(dense.n, dense.t_tab) ?
                @sprintf("wins %.2fx", mpps(dense.ngrid, dense.t_grid) / mpps(dense.n, dense.t_tab)) :
                @sprintf("loses %.2fx", mpps(dense.n, dense.t_tab) / mpps(dense.ngrid, dense.t_grid)))
    @printf("       production  (over=8): FFT+linear      %6.1f vs direct+table %6.1f  -> direct %.2fx\n",
            mpps(prod.n, prod.t_fft), mpps(prod.n, prod.t_tab),
            prod.t_fft / prod.t_tab)
end
println("""
Read this as: FFTW is *not* slow — per grid point it is competitive with, and at
large m faster than, direct summation, which is the intuition that motivated the
FFT-correlation design in the first place.  What sinks it in this application is
that it cannot deliver those grid points where they are wanted.  Two costs,
both visible above:

  * the fine grid must be `numbetween` times finer than the trial spacing for
    the linear interpolation to be accurate, so 8 of every 9 grid points at the
    production settings are computed and discarded (`over` column); and
  * the linear interpolation that reads them is itself not free, and is an
    approximation — see `err_fft`, which is ~1e-2 at the production settings
    against the direct path's ~1e-15.

The FFT only comes out ahead in the corner where the grid is exactly as dense as
the points wanted *and* the kernel is wide (large m).  The search cannot use that
corner: the grid is anchored to integer Fourier bins while the trials sit at an
arbitrary sub-bin offset, so without a per-chunk kernel rebuild (a third
transform) linear interpolation is unavoidable, and that is what forces the
oversampling.""")

# --- plots ------------------------------------------------------------------

const C_REF  = RGBf(0.55, 0.55, 0.58)
const C_DIR  = RGBf(0.85, 0.37, 0.13)
const C_TAB  = RGBf(0.75, 0.15, 0.42)
const C_FFT  = RGBf(0.15, 0.42, 0.78)
const C_GRID = RGBf(0.45, 0.70, 0.90)

fig = Figure(size=(1150, 470))

# Panel 1: throughput vs m, at the production oversampling and at a dense grid.
for (col, nb) in enumerate((16, 2))
    sel = filter(r -> r.nb == nb, rows)
    ms = [r.m for r in sel]
    ax = Axis(fig[1, col], xscale=log2, yscale=log10,
              xticks=([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"]),
              yticks=([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"]),
              xlabel="kernel width m", ylabel="M interpolated points / s",
              title = nb == 16 ?
                  "Production grid: numbetween=16, step=0.5 (8x oversampled)" :
                  "Dense grid: numbetween=2, step=0.5 (1x — no wasted grid points)")
    lines!(ax, ms, [mpps(r.n, r.t_dir) for r in sel], color=C_DIR, linewidth=2.5)
    scatter!(ax, ms, [mpps(r.n, r.t_dir) for r in sel], color=C_DIR, markersize=11,
             label="direct (exact)")
    lines!(ax, ms, [mpps(r.n, r.t_fft) for r in sel], color=C_FFT, linewidth=2.5)
    scatter!(ax, ms, [mpps(r.n, r.t_fft) for r in sel], color=C_FFT, marker=:rect,
             markersize=11, label="FFT + linear interp")
    lines!(ax, ms, [mpps(r.ngrid, r.t_grid) for r in sel], color=C_GRID,
           linewidth=2, linestyle=:dash)
    scatter!(ax, ms, [mpps(r.ngrid, r.t_grid) for r in sel], color=C_GRID,
             marker=:utriangle, markersize=10, label="FFT, per grid point")
    lines!(ax, ms, [mpps(r.n, r.t_ref) for r in sel], color=C_REF, linewidth=1.5,
           linestyle=:dot)
    scatter!(ax, ms, [mpps(r.n, r.t_ref) for r in sel], color=C_REF, marker=:xcross,
             markersize=9, label="fourier_interp (per-point reference)")
    tab = filter(r -> !isnan(r.t_tab), sel)
    if !isempty(tab)
        scatter!(ax, [r.m for r in tab], [mpps(r.n, r.t_tab) for r in tab],
                 color=C_TAB, marker=:diamond, markersize=14,
                 label="direct + phase table (production)")
    end
    col == 1 && axislegend(ax, position=:lb, framevisible=false, labelsize=10)
end
save(joinpath(@__DIR__, "interp_bench_throughput.png"), fig)

# Panel 2: the crossover — direct/FFT speedup vs oversampling, per m.
fig2 = Figure(size=(640, 470))
ax2 = Axis(fig2[1, 1], xscale=log2, yscale=log2,
           xticks=([1, 2, 4, 8, 16], ["1x", "2x", "4x", "8x", "16x"]),
           yticks=([0.25, 0.5, 1, 2, 4, 8, 16],
                   ["0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x"]),
           xlabel="oversampling (fine-grid points per requested point)",
           ylabel="direct / (FFT + linear)  speedup",
           title="Where direct summation beats FFT correlation (N = 2048)",
           subtitle="solid: production direct+phase-table   faint: direct recomputing weights per point")
for (i, m) in enumerate((8, 16, 32, 64, 128))
    sel = filter(r -> r.m == m, rows)
    xs = [r.over for r in sel]
    c = RGBf(0.1 + 0.18 * (i - 1), 0.35, 0.85 - 0.15 * (i - 1))
    # The untabulated variant, for reference: same algorithm, m divisions per
    # point instead of a table lookup.
    lines!(ax2, xs, [r.t_fft / r.t_dir for r in sel], color=(c, 0.3),
           linewidth=1.5, linestyle=:dash)
    ys = [r.t_fft / r.t_tab for r in sel]
    lines!(ax2, xs, ys, color=c, linewidth=2.5)
    scatter!(ax2, xs, ys, color=c, markersize=11, label="m = $m")
end
hlines!(ax2, [1.0], color=:black, linestyle=:dash, linewidth=1.5)
text!(ax2, 1.1, 1.05, text="FFT faster below, direct faster above", fontsize=10,
      align=(:left, :bottom))
vlines!(ax2, [8.0], color=RGBf(0.5, 0.5, 0.5), linestyle=:dot, linewidth=1.5)
text!(ax2, 8.3, 0.6, text="production\n(nb=16, step=0.5)", fontsize=10,
      align=(:left, :bottom))
axislegend(ax2, position=:lt, framevisible=false, labelsize=10)
save(joinpath(@__DIR__, "interp_bench_crossover.png"), fig2)

println("\nWrote bench/interp_bench_throughput.png and bench/interp_bench_crossover.png")
