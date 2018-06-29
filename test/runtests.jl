using Particles
using Base.Test

@testset "Particles" begin
    include("component.jl")
    include("statepriors.jl")
    include("particle.jl")
    include("fearnhead.jl")
    include("nix2.jl")
    include("niw.jl")
end
