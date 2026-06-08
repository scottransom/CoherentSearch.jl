# Readers for PRESTO-style `.inf` metadata and `.fft` amplitude files.
# Ports `coherent_search.utils` (simpleinf, fftfile).

using Mmap

"""
    SimpleInf

A minimal PRESTO `.inf` reader holding only the fields this search needs.
Mirrors the Python `simpleinf` class (key params only).
"""
struct SimpleInf
    path::String
    object::Union{String,Nothing}
    epoch::Union{Float64,Nothing}
    N::Union{Int,Nothing}            # number of bins in the time series
    dt::Union{Float64,Nothing}       # width of each time-series bin (s)
    DM::Union{Float64,Nothing}
end

_inf_value(line) = strip(split(line, "=")[end])

function SimpleInf(path::AbstractString)
    object = nothing; epoch = nothing; N = nothing; dt = nothing; DM = nothing
    isfile(path) || error("The .inf file '$path' was not found.")
    for line in eachline(path)
        if startswith(line, " Object being observed")
            object = String(_inf_value(line))
        elseif startswith(line, " Epoch")
            epoch = parse(Float64, _inf_value(line))
        elseif startswith(line, " Number of bins")
            N = parse(Int, _inf_value(line))
        elseif startswith(line, " Width of each time series bin")
            dt = parse(Float64, _inf_value(line))
        elseif startswith(line, " Dispersion measure")
            DM = parse(Float64, _inf_value(line))
        end
    end
    return SimpleInf(String(path), object, epoch, N, dt, DM)
end

"""
    FFTFile

A memory-mapped PRESTO `.fft` file plus the metadata derived from its `.inf`.
Mirrors the Python `fftfile` class.

The amplitudes are stored as `ComplexF32`.  As in PRESTO, element 1 packs the
DC term in its real part and the Nyquist term in its imaginary part.
"""
struct FFTFile
    path::String
    amps::Vector{ComplexF32}     # 1-based; amps[1] = DC.real + Nyquist.imag*im
    inf::SimpleInf
    N::Int
    T::Float64
    df::Float64
    dereddened::Bool
    detrended::Bool
    DC::Float32
    Nyquist::Float32
end

function FFTFile(path::AbstractString)
    inf_path = string(chop(String(path); tail=4), ".inf")  # strip ".fft"
    inf = SimpleInf(inf_path)
    inf.N === nothing && error("Missing 'Number of bins' in $inf_path")
    inf.dt === nothing && error("Missing 'Width of each time series bin' in $inf_path")
    amps = open(path, "r") do io
        Mmap.mmap(io, Vector{ComplexF32}, (filesize(path) ÷ sizeof(ComplexF32),))
    end
    N = inf.N
    T = N * inf.dt
    dereddened = occursin("_red.fft", String(path))
    return FFTFile(String(path), amps, inf, N, T, 1.0 / T,
                   dereddened, dereddened,
                   real(amps[1]), imag(amps[1]))
end

"""
    freqs(ft::FFTFile) -> range

The Fourier frequencies (Hz) corresponding to bins `0 .. N÷2-1`.
"""
freqs(ft::FFTFile) = range(0.0, step=ft.df, length=ft.N ÷ 2)
