using Particles
using Base.Test

@testset "Particles" begin
    include("nix2.jl")
    include("particle.jl")
    include("fearnhead.jl")
end
