using Particles
using Test
using Random

@testset "Particles" begin
    include("component.jl")
    include("statepriors.jl")
    include("particle.jl")
    include("fearnhead.jl")
    include("nix2.jl")
    include("niw.jl")
end
