# not run during tests, but reproduce teh effective sample size simulations from
# Fearnhead (2004).
addprocs(11)

@everywhere begin
    using ParallelDataTransfer
    using Particles
    using Particles: ncomponents
    using Distributions
    using Distributions: MixtureModel
    using ConjugatePriors: NormalInverseGamma, NormalInverseChisq

    function m_m2(ps::FearnheadParticles)
        ns = ncomponents.(ps.particles)
        ws = weight.(ps.particles)
        ws ./= sum(ws)
        (ns ⋅ ws, ns.^2 ⋅ ws)
    end

    function m_m2(gc::GibbsCRP, n; burnin=500)
        for _ in 1:burnin
            sample!(gc)
        end
        m = 0
        m2 = 0
        for _ in 1:n
            sample!(gc)
            ncomp = length(gc.components) - length(gc.empties)
            m += ncomp
            m2 += ncomp^2
        end
        (m / n, m2 / n)
    end

    n_samp = 5000
    n_dat = 200
    α = 0.5
    prior = NormalInverseChisq(NormalInverseGamma(0., 25., 1., 1.))

end

function ess(f::Function; n::Int=100)
    sendto(workers(), f=f)
    ms = SharedArray{Float64}(n)
    m2s = SharedArray{Float64}(n)
    @sync @parallel for i in 1:n
        srand(i)
        ms[i], m2s[i] = f()
    end
    mbar = mean(ms)
    return (mean(m2s) - mbar^2) / mean((ms .- mbar).^2)
end

results = []
for σ in [0.5, 2.5]
    for μ in [0.5, 1, 2, 5]
        mm = MixtureModel([Normal(0, 0.5), Normal(μ, 0.5), Normal(2μ, σ)],
                          [1/2, 1/6, 1/3])
        x = rand(mm, n_dat)
        fc = ess(() -> m_m2(fit!(FearnheadParticles(n_samp, prior, α), x, false)))
        gs = ess(() -> m_m2(GibbsCRP(prior, α, x), n_samp, burnin=500))
        @show μ, σ, fc, gs
        push!(results, (μ, σ, fc, gs))
    end
end

# (μ, σ, fc, gs) = (0.5, 0.5, 707.4197999272996, 250.4401078254187)
# (μ, σ, fc, gs) = (1.0, 0.5, 4.986123628551047, 231.97429021977032)
# (μ, σ, fc, gs) = (2.0, 0.5, 24.706860608800266, 712.9339627024202)
# (μ, σ, fc, gs) = (5.0, 0.5, 2294.7314794665976, 517.5777000665694)
# (μ, σ, fc, gs) = (0.5, 2.5, 3.6015506679644673, 192.154516933825)
# (μ, σ, fc, gs) = (1.0, 2.5, 4.093719484784503, 317.6933662619361)
# (μ, σ, fc, gs) = (2.0, 2.5, 3.3444126682400066, 310.6637042075222)
# (μ, σ, fc, gs) = (5.0, 2.5, 9.241825755336542, 482.340345642232)
