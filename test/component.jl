using ConjugatePriors, Distributions
using Distributions: NormalStats

@testset "Component" begin
    @testset "add/sub from NormalStats" begin
        using Particles: add, sub
        @test reduce(add, NormalStats(0,0,0,0), 1:10) == suffstats(Normal, 1:10)
        @test reduce(sub, suffstats(Normal, 1:10), 1:5) == suffstats(Normal, 6:10)
        @test sub(suffstats(Normal, [1]), 1) == NormalStats(0,0,0,0)
    end
end
