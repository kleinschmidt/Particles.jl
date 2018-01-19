module Particles

using
    Distributions,
    ConjugatePriors,
    StatsBase,
    ArgCheck

import StatsBase: fit!, fit

using Distributions: NormalStats
using ConjugatePriors: posterior_canon, NormalInverseChisq

export
    FearnheadParticles,
    Particle,
    InfiniteParticle,
    Component,
    NormalInverseChisq,
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
    marginal_log_posterior
#     normalize_clusters!,


const DEBUG = false

macro debug(msg)
    DEBUG ? :(println(string($(esc(msg))))) : nothing
end

include("component.jl")
include("particle2.jl")
include("fearnhead.jl")

end # module
