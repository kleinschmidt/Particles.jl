module Particles

using
    LinearAlgebra,
    Random,
    SpecialFunctions,
    Statistics

using
    Distributions,
    PDMats,
    ConjugatePriors,
    StatsBase,
    StatsFuns

import StatsBase: fit!, fit

using Distributions: NormalStats, MvNormalStats, GenericMvTDist
using ConjugatePriors: posterior_canon, NormalInverseChisq, NormalInverseWishart
using StatsBase: Weights
using StatsFuns: logmvgamma, logπ

export
    ParticleFilter,
    FearnheadParticles,
    ChenLiuParticles,
    InfiniteParticle,
    Component,
    NormalInverseChisq,
    GibbsCRP,
    GibbsCRPSamples,
    ChineseRestaurantProcess,
    StickyCRP,
    ChangePoint,
    NStatePrior,
    Labeled,
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
    log_prior,
    marginal_log_prior,
    simulate,
    ncomponents_dist,
    sample!,
    state_entropy,
    randindex

include("component.jl")
include("statepriors.jl")
include("particle.jl")
include("filters.jl")
include("fearnhead.jl")
include("chenliu.jl")
include("gibbs.jl")
include("labeled.jl")

end # module
