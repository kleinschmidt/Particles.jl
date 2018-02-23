using Distributions
using ConjugatePriors: NormalInverseWishart

@testset "NormalInverseWishart" begin

    @testset "Posterior predictive" begin
        d = NormalInverseWishart([0., 1.], 10., [4. 2.; 2. 3.], 11.)
        d_pp_samps = zeros(2, 1_000_000)
        for i in 1:1_000_000
            d_pp_samps[:,i] = rand(MvNormal(rand(d)...))
        end
        # d_pp_samps = mapreduce(i->rand(MvNormal(rand(d)...)), append!, Float64[], 1:1_000_000)
        pp_samps = rand(posterior_predictive(d), 1_000_000)

        isapprox(cov(d_pp_samps'), cov(pp_samps'), atol=0.01)
        isapprox(mean(d_pp_samps, 2), mean(pp_samps, 2), atol=0.01)
    end

end
