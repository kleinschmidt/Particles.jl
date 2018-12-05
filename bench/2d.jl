using BenchmarkTools,
    Particles,
    Distributions,
    ConjugatePriors,
    StaticArrays,
    Compat,
    Random

using ConjugatePriors: NormalInverseWishart
using Compat.LinearAlgebra

Random.seed!(100)

truth = MixtureModel([MvNormal(-2 .* ones(2), Matrix(1.0I,2,2)),
                      MvNormal(2 .* ones(2), Matrix(1.0I,2,2))])
y = [rand(truth) for _ in 1:100]

prior = NormalInverseWishart(zeros(2), 0.1, Matrix(1.0I,2,2), 3.0)

function f(y, prior)
    stateprior = ChineseRestaurantProcess(1.)
    ps = ChenLiuParticles(100, prior, stateprior)
    filter!(ps, y, false)
end

f(y, prior)
@btime(f($y, $prior))

sprior = NormalInverseWishart(SVector{2}(zeros(2)), 0.1, cholesky(SMatrix{2,2}(1.0I)), 3.0)

f(y, sprior)
@btime(f($y, $sprior))

sy = SVector{2}.(y)
f(sy, sprior)
ps3 = @btime f($sy, $sprior)
