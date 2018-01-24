using Particles
using Base.Test

@testset "Particles" begin
    include("component.jl")
    include("particle.jl")
    include("fearnhead.jl")
end
