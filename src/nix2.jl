"""
    FitNormalInverseChisq <: ExactStat{0}

An `OnlineStat` that fits a normal distribution with unknown mean and variance,
using a conjugate `NormalInverseChisq` prior.

# Constructors

```julia
FitNormalInverseChisq(μ0::T, σ20::T, κ0::T, ν0::T) where T<:Real
FitNormalInverseChisq(ps::NTuple{4,T}) where {T<:Real}
# uninformative prior
FitNormalInverseChisq()
```

# Examples

```julia
f = FitNormalInverseChisq()
fit!(Series(f), randn(100))
posterior = NormalInverseChisq(f)
predictive = posterior_predictive(posterior)
```

"""

mutable struct FitNormalInverseChisq <: ExactStat{0}
    var::Variance
    prior::NormalInverseChisq
    FitNormalInverseChisq(prior::NormalInverseChisq) = new(Variance(), prior)
end

FitNormalInverseChisq(μ0::T, σ20::T, κ0::T, ν0::T) where T<:Real =
    FitNormalInverseChisq(NormalInverseChisq(μ0, σ20, κ0, ν0))
FitNormalInverseChisq(ps::NTuple{4,T}) where {T<:Real} = FitNormalInverseChisq(ps...)
FitNormalInverseChisq() = FitNormalInverseChisq(0.0, 1.0, 0.0, 0.0)

OnlineStats.nobs(o::FitNormalInverseChisq) = nobs(o.var)

## Fitting
Base.convert(::Type{NormalStats}, o::Variance) =
    NormalStats(o.nobs*o.μ, o.μ, o.nobs*o.σ2, Float64(o.nobs))
fit!(o::FitNormalInverseChisq, x, w::Float64) = (fit!(o.var, x, w); o)
fit!(o::FitNormalInverseChisq, x) = fit!(o, x, 1.0/(nobs(o.var)+1))
NormalInverseChisq(o::FitNormalInverseChisq) = posterior_canon(o.prior, NormalStats(o.var))
function OnlineStatsBase._value(o::FitNormalInverseChisq)
    post =
        nobs(o.var) > 0 ?
        params(NormalInverseChisq(o)) :
        params(o.prior)
end

posterior_predictive(d::NormalInverseChisq) =
    LocationScale(d.μ, sqrt((1+d.κ)*d.σ2/d.κ), TDist(d.ν))
posterior_predictive(f::FitNormalInverseChisq) = posterior_predictive(NormalInverseChisq(f))

"""
    marginal_lhood(o::FitNormalInverseChisq)

The marginal likelihood of data fit so far by `o`, which is the integral of the
likelihood given the mean and variance under the prior on those parameters.
"""
function marginal_lhood(o::FitNormalInverseChisq)
    μ0, σ20, κ0, ν0 = params(o.prior)
    μn, σ2n, κn, νn = params(NormalInverseChisq(o))
    n = νn - ν0
    gamma(νn*0.5)/gamma(ν0*0.5) * 
        sqrt(κ0/κn) * 
        (ν0*σ20)^(ν0*0.5) * (νn*σ2n)^(-νn*0.5) / 
        π^(n*0.5)
end

"""
    marginal_log_lhood(o::FitNormalInverseChisq)

The marginal likelihood of data fit so far by `o`, which is the integral of the
likelihood given the mean and variance under the prior on those parameters.
"""
function marginal_log_lhood(o::FitNormalInverseChisq)
    μ0, σ20, κ0, ν0 = params(o.prior)
    μn, σ2n, κn, νn = params(NormalInverseChisq(o))
    n = νn - ν0
    lgamma(νn*0.5) - lgamma(ν0*0.5) +
        0.5*(log(κ0)-log(κn)) +
        (0.5*ν0)*(log(ν0)+log(σ20)) - (0.5*νn)*(log(νn)+log(σ2n)) -
        (0.5*n)*log(π)
end
