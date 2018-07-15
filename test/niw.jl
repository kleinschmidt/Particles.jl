using Distributions
using ConjugatePriors: NormalInverseWishart, posterior_canon
using Particles: marginal_lhood, marginal_log_lhood

ConjugatePriors.NormalInverseWishart(nix2::NormalInverseChisq) =
    NormalInverseWishart([nix2.μ], nix2.κ, nix2.ν*reshape([nix2.σ2], 1, 1), nix2.ν)


@testset "NormalInverseWishart" begin

    @testset "Posterior predictive" begin
        d = NormalInverseWishart([0., 1.], 10., [4. 2.; 2. 3.], 11.)
        d_pp_samps = zeros(2, 1_000_000)
        for i in 1:1_000_000
            d_pp_samps[:,i] = rand(MvNormal(rand(d)...))
        end
        pp_samps = rand(posterior_predictive(d), 1_000_000)

        isapprox(cov(d_pp_samps'), cov(pp_samps'), atol=0.01)
        isapprox(mean(d_pp_samps, 2), mean(pp_samps, 2), atol=0.01)
    end

    @testset "Marginal likelihood" begin
        srand(1001)

        prior = NormalInverseWishart([0., 1.], 10., eye(2), 50.)
        x = rand(MvNormal(rand(prior)...), 10)
        ss = suffstats(MvNormal, x)
        # this is trivially true at the moment since one just calls the other
        @test marginal_log_lhood(prior, ss) ≈ log(marginal_lhood(prior, ss))
        xvecs = [x[:,i] for i in 1:size(x,2)]
        logpdfs = [sum(logpdf.(MvNormal(prior_samp...), xvecs))
                   for prior_samp
                   in (rand(prior) for _ in 1:1000)]
        @test isapprox(marginal_log_lhood(prior, ss), log(mean(exp.(logpdfs))), rtol=0.1)
        # sanity check: lhood under samples from prior is more variable than
        # error from analytical expression
        @test std(logpdfs) > abs(marginal_log_lhood(prior, ss) - log(mean(exp.(logpdfs))))
    end

    # I think the parameter space is too high-dimensional to have any hope of
    # testing reliably with sampling like this.  Next best is to check against
    # the 1-D case which is equivalent to the Normal-Inverse Chi-squared
    @testset "Equivalence with Normal-Inverse Chi-squared" begin
        nix2 = NormalInverseChisq(1., 2., 3., 4.)
        niw = NormalInverseWishart(nix2)

        x = rand(Normal(3, 2), 100)

        ss = suffstats(Normal, x)
        ss_mv = suffstats(MvNormal, reshape(x, 1, :))

        post_nix2 = posterior_canon(nix2, ss)
        post_niw = posterior_canon(niw, ss_mv)

        @test all(post_nix2.μ .≈ post_niw.mu)
        @test post_nix2.κ ≈ post_niw.kappa
        @test post_nix2.ν ≈ post_niw.nu
        @test all(post_nix2.σ2 .≈ full(post_niw.Lamchol)[1] ./ post_niw.nu)

        μ, σ2 = rand(post_nix2)
        @test logpdf(post_nix2, μ, σ2) ≈ logpdf(post_niw, [μ], reshape([σ2], 1, 1))
        @test logpdf(nix2, μ, σ2) ≈ logpdf(niw, [μ], reshape([σ2], 1, 1))

        @test marginal_log_lhood(nix2, ss) ≈ marginal_log_lhood(niw, ss_mv)

    end
end


