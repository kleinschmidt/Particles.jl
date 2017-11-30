module Particles

using
    Distributions,
    ConjugatePriors,
    OnlineStatsBase,
    OnlineStats,
    StatsBase

import StatsBase: fit!

using Distributions: NormalStats
using ConjugatePriors: posterior_canon, NormalInverseChisq
using OnlineStats: Variance, nobs, EqualWeight
using OnlineStatsBase: ExactStat

export
    FitNormalInverseChisq,
    Particle,
    FearnheadParticles,
    fit!,
    posterior_predictive,
    marginal_lhood,
    marginal_log_lhood,
    putatives


const DEBUG = true

macro debug(msg)
    DEBUG ? :(println(string($(esc(msg))))) : nothing
end


# package code goes here
include("nix2.jl")
include("particle.jl")
include("fearnhead.jl")

end # module
