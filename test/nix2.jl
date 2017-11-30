using OnlineStats
using Distributions
using ConjugatePriors: posterior_canon, NormalInverseChisq

@testset "Normal-χ^-2" begin
    x = rand(Normal(3, 5), 100)
    o = FitNormalInverseChisq()
    s = Series(o)

    @testset "Fitting" begin

        for xi in x
            fit!(s, xi)
        end

        Base.:≈(d1::NormalInverseChisq, d2::NormalInverseChisq) = all(params(d1) .≈ params(d2))
        @test NormalInverseChisq(o) ≈ posterior_canon(o.prior, suffstats(Normal, x))

    end

    @testset "Posterior predictive" begin
        pp = posterior_predictive(NormalInverseChisq(o))

        xpp = rand(posterior_predictive(NormalInverseChisq(o)), 1_000_000)
        xpp2 = [rand(Normal(μ, sqrt(σ2))) for (μ, σ2) in (rand(NormalInverseChisq(o)) for _ in 1:1_000_000)]

        @test isapprox(mean(xpp), mean(xpp2), rtol=0.01)
        @test isapprox(std(xpp), std(xpp2), rtol=0.01)
    end

    @testset "Marginal likelihood" begin
        o2 = FitNormalInverseChisq(0.0, 1.0, 1.0, 1.0)
        fit!(Series(o2), x)
        @test marginal_log_lhood(o2) ≈ log(marginal_lhood(o2))
    end
    
end
