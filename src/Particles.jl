module Particles

using
    Distributions,
    ConjugatePriors,
    StatsBase,
    ArgCheck

import StatsBase: fit!, fit

using Distributions: NormalStats
using ConjugatePriors: posterior_canon, NormalInverseChisq
using StatsBase: Weights

export
    ParticleFilter,
    FearnheadParticles,
    ChenLiuParticles,
    Particle,
    InfiniteParticle,
    Component,
    NormalInverseChisq,
    GibbsCRP,
    fit,
    fit!,
    putatives,
    weight,
    nobs,
    assignments,
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
include("particle2.jl")
include("filters.jl")
include("fearnhead.jl")
include("chenliu.jl")
include("gibbs.jl")

end # module
