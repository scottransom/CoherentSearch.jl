using Test

@testset "CoherentSearch.jl" begin
    include("test_fourierinterp.jl")
    include("test_fileio.jl")
    include("test_search.jl")
    include("test_candidate.jl")
    include("test_cli.jl")
    include("test_toy.jl")
    # Skips itself unless a functional CUDA backend is present; CUDA is a weak
    # dependency, so an ordinary Pkg.test() never has one.
    include("test_gpu.jl")
end
