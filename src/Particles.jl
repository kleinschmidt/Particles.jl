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
    Particle,
    Component,
    NormalInverseChisq,
    fit,
    putatives
#     # InfiniteParticle,
#     # FearnheadParticles,
#     fit,
#     # posterior_predictive,
#     marginal_lhood,
#     marginal_log_lhood,
#     marginal_posterior,
#     marginal_log_posterior,
#     putatives,
#     normalize_clusters!,
#     weight


const DEBUG = false

macro debug(msg)
    DEBUG ? :(println(string($(esc(msg))))) : nothing
end


# package code goes here
include("component.jl")
include("particle2.jl")
# include("nix2.jl")
# include("particle.jl")
# include("fearnhead.jl")

end # module
