using BenchmarkTools,
    Particles,
    Distributions,
    ConjugatePriors,
    StaticArrays,
    LinearAlgebra,
    Random

using ConjugatePriors: NormalInverseWishart

Random.seed!(100)

truth = MixtureModel([MvNormal(-2 .* ones(2), Matrix(1.0I,2,2)),
                      MvNormal(2 .* ones(2), Matrix(1.0I,2,2))])
y = [rand(truth) for _ in 1:100]

prior = NormalInverseWishart(zeros(2), 0.1, Matrix(1.0I,2,2), 3.0)

function f(y, prior)
    stateprior = ChineseRestaurantProcess(1.)
    ps = FearnheadParticles(100, prior, stateprior)
    filter!(ps, y, false)
end

f(y, prior)
@btime(f($y, $prior))


sprior = NormalInverseWishart(SVector{2}(zeros(2)), 0.1,
                              cholesky(SMatrix{2,2}(1.0I)), 3.0)

f(y, sprior)
@btime(f($y, $sprior))


sy = SVector{2}.(y);
f(sy, sprior);
ps3 = @btime f($sy, $sprior)


## faa9976:
##  303.332 ms (2622263 allocations: 180.38 MiB)
##  124.002 ms (1403079 allocations: 110.79 MiB)
##   81.780 ms (821809 allocations: 66.29 MiB)

