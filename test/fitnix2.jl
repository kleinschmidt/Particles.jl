using OnlineStats
using Distributions
using ConjugatePriors: posterior_canon, NormalInverseChisq

@testset "Normal-χ^-2" begin
    x = rand(Normal(3, 5), 100)
    o = FitNormalInverseChisq()
    s = Series(o)

    for xi in x
        fit!(s, xi)
    end

    Base.:≈(d1::NormalInverseChisq, d2::NormalInverseChisq) = all(params(d1) .≈ params(d2))
    @test NormalInverseChisq(o) ≈ posterior_canon(o.prior, suffstats(Normal, x))
end
