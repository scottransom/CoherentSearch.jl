using Test

@testset "CoherentSearch.jl" begin
    include("test_fourierinterp.jl")
    include("test_fileio.jl")
    include("test_search.jl")
end
