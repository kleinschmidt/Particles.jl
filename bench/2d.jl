using BenchmarkTools
using Particles, Distributions, ConjugatePriors
using ConjugatePriors: NormalInverseWishart
using Compat
using Compat.LinearAlgebra


truth = MixtureModel([MvNormal(-2 .* ones(2), Matrix(1.0I,2,2)),
                      MvNormal(2 .* ones(2), Matrix(1.0I,2,2))])
y = [rand(truth) for _ in 1:100]

function f(y)
    ps = ChenLiuParticles(100,
                          NormalInverseWishart(zeros(2), 0.1, Matrix(1.0I,2,2), 3.0),
                          ChineseRestaurantProcess(1.))
    filter!(ps, y, false)
end

f(y)
@btime(f($y))
