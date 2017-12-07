module Particles

using
    Distributions,
    ConjugatePriors,
    OnlineStatsBase,
    OnlineStats,
    StatsBase,
    ArgCheck

import StatsBase: fit!

using Distributions: NormalStats
using ConjugatePriors: posterior_canon, NormalInverseChisq
using OnlineStats: Variance, nobs, EqualWeight
using OnlineStatsBase: ExactStat

export
    Series,
    FitNormalInverseChisq,
    Particle,
    InfiniteParticle,
    FearnheadParticles,
    fit!,
    posterior_predictive,
    marginal_lhood,
    marginal_log_lhood,
    putatives,
    normalize_clusters!


const DEBUG = true

macro debug(msg)
    DEBUG ? :(println(string($(esc(msg))))) : nothing
end


# package code goes here
include("nix2.jl")
include("particle.jl")
include("fearnhead.jl")

end # module
