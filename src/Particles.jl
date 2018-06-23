module Particles

using
    Distributions,
    ConjugatePriors,
    StatsBase,
    ArgCheck,
    StatsFuns

import StatsBase: fit!, fit

using Distributions: NormalStats, MvNormalStats
using ConjugatePriors: posterior_canon, NormalInverseChisq, NormalInverseWishart
using StatsBase: Weights
using StatsFuns: logmvgamma, logπ

export
    ParticleFilter,
    FearnheadParticles,
    ChenLiuParticles,
    Particle,
    InfiniteParticle,
    Component,
    NormalInverseChisq,
    GibbsCRP,
    GibbsCRPSamples,
    fit,
    fit!,
    putatives,
    weight,
    nobs,
    particles,
    assignments,
    assignment_similarity,
    posterior_predictive,
    marginal_lhood,
    marginal_log_lhood,
    marginal_posterior,
    marginal_log_posterior,
    ncomponents_dist,
    sample!
#     normalize_clusters!,


const DEBUG = false

macro debug(msg)
    DEBUG ? :(println(string($(esc(msg))))) : nothing
end

include("component.jl")
include("statepriors.jl")
include("particle.jl")
include("filters.jl")
include("fearnhead.jl")
include("chenliu.jl")
include("gibbs.jl")

end # module
