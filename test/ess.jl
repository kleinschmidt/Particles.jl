# not run during tests, but reproduce teh effective sample size simulations from
# Fearnhead (2004).
using Distributed
addprocs()

@everywhere begin
    using ParallelDataTransfer, SharedArrays
    using Particles, Distributions, StatsBase, Random
    using Particles: ncomponents
    using Distributions: MixtureModel
    using ConjugatePriors: NormalInverseGamma, NormalInverseChisq

    function m_m2(ps::ParticleFilter)
        ns = ncomponents.(particles(ps))
        ws = Weights(weight.(particles(ps)))
        (mean(ns, ws), var(ns, ws))
    end

    function m_m2(gc::GibbsCRP, n; burnin=500)
        for _ in 1:burnin
            sample!(gc)
        end
        ns = Int[]
        for _ in 1:n
            sample!(gc)
            ncomp = length(gc.components) - length(gc.empties)
            push!(ns, ncomp)
        end
        (mean(ns), var(ns))
    end

    n_samp = 5000
    n_dat = 200
    α = 0.5
    prior = convert(NormalInverseChisq, NormalInverseGamma(0., 25., 1., 1.))
    stateprior = ChineseRestaurantProcess(α)

end

function ess(f::Function; n::Int=100)
    sendto(workers(), f=f)
    ms = SharedArray{Float64}(n)
    m2s = SharedArray{Float64}(n)
    @sync @distributed for i in 1:n
        Random.seed!(i)
        ms[i], m2s[i] = f()
    end
    return mean(m2s) / var(ms)
end

n_ess = 100
results = []
for σ in [0.5, 2.5]
    for μ in [0.5, 1, 2, 5]
        mm = MixtureModel([Normal(0, 0.5), Normal(μ, 0.5), Normal(2μ, σ)],
                          [1/2, 1/6, 1/3])
        x = rand(mm, n_dat)
        fc = ess(() -> m_m2(filter!(FearnheadParticles(n_samp, prior, stateprior), x, false)), n=n_ess)
        cl = ess(() -> m_m2(filter!(ChenLiuParticles(n_samp, prior, stateprior, rejuv=50.), x, false)), n=n_ess)
        gs = ess(() -> m_m2(GibbsCRP(prior, α, x), n_samp, burnin=500), n=n_ess)
        @show μ, σ, fc, cl, gs
        push!(results, (μ, σ, fc, cl, gs))
    end
end

using JLD2
@save "ess_results_$(DateTime(now())).jld2" results 

# run using ESS formula from Fearnhead(2004)
# (μ, σ, fc, gs) = (0.5, 0.5, 707.4197999272996, 250.4401078254187)
# (μ, σ, fc, gs) = (1.0, 0.5, 4.986123628551047, 231.97429021977032)
# (μ, σ, fc, gs) = (2.0, 0.5, 24.706860608800266, 712.9339627024202)
# (μ, σ, fc, gs) = (5.0, 0.5, 2294.7314794665976, 517.5777000665694)
# (μ, σ, fc, gs) = (0.5, 2.5, 3.6015506679644673, 192.154516933825)
# (μ, σ, fc, gs) = (1.0, 2.5, 4.093719484784503, 317.6933662619361)
# (μ, σ, fc, gs) = (2.0, 2.5, 3.3444126682400066, 310.6637042075222)
# (μ, σ, fc, gs) = (5.0, 2.5, 9.241825755336542, 482.340345642232)

# run using mean(vars) / var(means)
# (μ, σ, fc, gs) = (0.5, 0.5, 117.49800834618993, 363.32269048299077)
# (μ, σ, fc, gs) = (1.0, 0.5, 6.220696850386848, 413.1589709680883)
# (μ, σ, fc, gs) = (2.0, 0.5, 19.496984108864652, 468.6555274028235)
# (μ, σ, fc, gs) = (5.0, 0.5, 4553.633313173106, 1009.8919191035734)
# (μ, σ, fc, gs) = (0.5, 2.5, 2.3728181956466163, 245.8546338400684)
# (μ, σ, fc, gs) = (1.0, 2.5, 1.751769156292295, 238.46679386421778)
# (μ, σ, fc, gs) = (2.0, 2.5, 3.9716813025135163, 373.4610563047813)
# (μ, σ, fc, gs) = (5.0, 2.5, 23.66274369175917, 321.8129868916317)

