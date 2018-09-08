using ConjugatePriors, Distributions
using ConjugatePriors: NormalInverseChisq, NormalInverseWishart
using Distributions: NormalStats, MvNormalStats
using Particles: add, sub, empty_suffstats

@testset "Component" begin
    @testset "add/sub from NormalStats" begin
        @test reduce(add, 1:10, init=NormalStats(0,0,0,0)) == suffstats(Normal, 1:10)
        @test reduce(sub, 1:5, init=suffstats(Normal, 1:10)) == suffstats(Normal, 6:10)
        @test sub(suffstats(Normal, [1]), 1) == NormalStats(0,0,0,0)
    end

    @testset "add/sub from MvNormalStats" begin
        Base.:(==)(a::MvNormalStats, b::MvNormalStats) =
            a.m == b.m &&
            a.s == b.s &&
            a.s2 == b.s2 &&
            a.tw == b.tw
        vecs = [[i,i+1] for i in 1.:10]
        @test reduce(add, vecs, init=MvNormalStats(zeros(2), zeros(2), zeros(2,2), 0)) ==
            suffstats(MvNormal, hcat(vecs...))
        @test reduce(sub, vecs[1:5], init=suffstats(MvNormal, hcat(vecs...))) ==
            suffstats(MvNormal, hcat(vecs[6:10]...))
        @test sub(suffstats(MvNormal, hcat(vecs[1])), vecs[1]) ==
            MvNormalStats(zeros(2), zeros(2), zeros(2,2), 0)
    end

    @testset "empty sufficient statistics" begin
        @test empty_suffstats(NormalInverseChisq()) == NormalStats(0, 0, 0, 0)
        ss1 = empty_suffstats(NormalInverseWishart(zeros(1), 1., ones(1,1), 1.))
        ss2 = MvNormalStats(zeros(1), zeros(1), zeros(1,1), 0.)
        @test all(getfield(ss1, n) == getfield(ss2, n) for n in fieldnames(typeof(ss1)))
        ss1 = empty_suffstats(NormalInverseWishart(zeros(3), 1., Matrix(1.0I,3,3), 1.))
        ss2 = MvNormalStats(zeros(3), zeros(3), zeros(3,3), 0.)
        @test all(getfield(ss1, n) == getfield(ss2, n) for n in fieldnames(typeof(ss1)))
    end
    
end
