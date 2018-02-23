using Distributions
using ConjugatePriors: posterior_canon, NormalInverseChisq
using Particles: marginal_lhood, marginal_log_lhood

@testset "Normal-χ^-2" begin
    x = rand(Normal(3, 5), 100)
    ss = suffstats(Normal, x)
    prior = NormalInverseChisq(0., 1.0, 2., 3.)
    posterior = posterior_canon(prior, ss)

    @testset "Posterior predictive" begin
        pp = posterior_predictive(posterior)

        xpp = rand(pp, 1_000_000)
        xpp2 = [rand(Normal(μ, sqrt(σ2))) for (μ, σ2) in (rand(posterior) for _ in 1:1_000_000)]

        @test isapprox(mean(xpp), mean(xpp2), rtol=0.01)
        @test isapprox(std(xpp), std(xpp2), rtol=0.01)
    end

    @testset "marginal likelihood" begin
        @test marginal_log_lhood(prior, ss) ≈ log(marginal_lhood(prior, ss))
        logpdfs = [sum(logpdf.(Normal(μ, sqrt(σ2)), x))
                   for (μ, σ2)
                   in (rand(prior) for _ in 1:100_000)]
        @test isapprox(marginal_log_lhood(prior, ss), log(mean(exp.(logpdfs))), rtol=0.001)
    end

end
