mutable struct FitNormalInverseChisq <: ExactStat{0}
    var::Variance
    prior::NormalInverseChisq
    FitNormalInverseChisq(prior::NormalInverseChisq) = new(Variance(), prior)
end

FitNormalInverseChisq(μ0::T, σ20::T, κ0::T, ν0::T) where T<:Real =
    FitNormalInverseChisq(NormalInverseChisq(μ0, σ20, κ0, ν0))
FitNormalInverseChisq(ps::NTuple{4,T}) where {T<:Real} = FitNormalInverseChisq(ps...)
FitNormalInverseChisq() = FitNormalInverseChisq(0.0, 1.0, 0.0, 0.0)

Base.convert(::Type{NormalStats}, o::Variance) = NormalStats(o.nobs*o.μ, o.μ, o.nobs*o.σ2, Float64(o.nobs))

fit!(o::FitNormalInverseChisq, x, w::Float64) = (fit!(o.var, x, w); o)
fit!(o::FitNormalInverseChisq, x) = (fit!(o.var, x); o)
NormalInverseChisq(o::FitNormalInverseChisq) = posterior_canon(o.prior, NormalStats(o.var))
function OnlineStatsBase._value(o::FitNormalInverseChisq)
    post =
        nobs(o.var) > 0 ?
        params(NormalInverseChisq(o)) :
        params(o.prior)
end
