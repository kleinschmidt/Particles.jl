using BenchmarkTools
using Particles, Distributions, ConjugatePriors
using ConjugatePriors: NormalInverseChisq

truth = MixtureModel([Normal(-2.), Normal(2.)])
y = rand(truth, 100)

function f(y)
    ps = ChenLiuParticles(100,
                          NormalInverseChisq(0., 1., 0.1, 1.0),
                          ChineseRestaurantProcess(1.))
    filter!(ps, y, false)
end

f(y)
@btime(f($y))
