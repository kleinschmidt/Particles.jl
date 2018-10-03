using BenchmarkTools
using Particles, Distributions, ConjugatePriors
using ConjugatePriors: NormalInverseWishart
using StaticArrays
using Compat
using Compat.LinearAlgebra


truth = MixtureModel([MvNormal(-2 .* ones(2), Matrix(1.0I,2,2)),
                      MvNormal(2 .* ones(2), Matrix(1.0I,2,2))])
y = [rand(truth) for _ in 1:100]

function f(y)
    prior = NormalInverseWishart(zeros(2), 0.1, Matrix(1.0I,2,2), 3.0)
    stateprior = ChineseRestaurantProcess(1.)
    ps = ChenLiuParticles(100, prior, stateprior)
    filter!(ps, y, false)
end

f(y)
@btime(f($y))



function g(y)
    prior = NormalInverseWishart(zeros(2), 0.1, cholesky(SMatrix{2,2}(1.0I)), 3.0)
    stateprior = ChineseRestaurantProcess(1.)
    ps = ChenLiuParticles(100, prior, stateprior)
    filter!(ps, y, false)
end

gps = g(y)
@btime(g($y))


sy = SVector{2}.(y)
g(sy)
ps3 = @btime g($sy)
