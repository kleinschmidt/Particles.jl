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
    FitNormalInverseChisq

# package code goes here
include("fitnix2.jl")

end # module
