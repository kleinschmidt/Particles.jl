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
    fit!,
    posterior_predictive,
    marginal_lhood,
    marginal_log_lhood

# package code goes here
include("nix2.jl")

end # module
