struct Component{P,S}
    prior::P
    suffstats::S
end

Component(prior::NormalInverseChisq) = Component(prior, NormalStats(0,0,0,0))
Component(params::NTuple{N,<:Real}) where N = Component(NormalInverseChisq(params...))

Base.show(io::IO, c::Component{P,S}) where {P,S} = print(io, "$P$(params(c)) w/ n=$(nobs(c))")

posterior_predictive(gc::Component) =
    posterior_predictive(posterior_canon(gc.prior, gc.suffstats))

posterior_predictive(d::NormalInverseChisq) =
    LocationScale(d.μ, sqrt((1+d.κ)*d.σ2/d.κ), TDist(d.ν))

Distributions.logpdf(c::Component, x) = logpdf(posterior_predictive(c), x)
Distributions.pdf(c::Component, x) = logpdf(posterior_predictive(c), x)

Distributions.params(c::Component) = params(posterior_canon(c.prior, c.suffstats))


# struct NormalStats <: SufficientStats
#     s::Float64    # (weighted) sum of x
#     m::Float64    # (weighted) mean of x
#     s2::Float64   # (weighted) sum of (x - μ)^2
#     tw::Float64    # total sample weight
# end

# based on teh wikipedia algorithm for online variance/mean
function add(ss::NormalStats, x)
    tw = ss.tw+1
    Δ = x - ss.m
    m = ss.m + Δ/tw
    s2 = tw>1 ? ss.s2 + Δ*(x-m) : 0
    return NormalStats(ss.s+x, m, s2, tw)
end

function sub(ss::NormalStats, x)
    tw = ss.tw-1
    if tw ≤ 0
        return NormalStats(0,0,0,0)
    else
        Δ = x - ss.m
        m = ss.m - Δ/tw
        s2 = ss.s2 - Δ*(x-m)
        return NormalStats(ss.s-x, m, s2, tw)
    end
end

add(c::Component, x) = Component(c.prior, add(c.suffstats, x))
sub(c::Component, x) = Component(c.prior, sub(c.suffstats, x))

nobs(c::Component) = Int(c.suffstats.tw)
Base.isempty(c::Component) = nobs(c) == 0


"""
    marginal_lhood(c::Component)

The marginal likelihood of data fit so far by `o`, which is the integral of the
likelihood given the mean and variance under the prior on those parameters.
"""
function marginal_lhood(c::Component{NormalInverseChisq{Float64}, NormalStats})
    μ0, σ20, κ0, ν0 = params(c.prior)
    μn, σ2n, κn, νn = params(posterior_canon(c.prior, c.suffstats))
    n = νn - ν0
    gamma(νn*0.5)/gamma(ν0*0.5) *
        sqrt(κ0/κn) *
        (ν0*σ20)^(ν0*0.5) * (νn*σ2n)^(-νn*0.5) /
        π^(n*0.5)
end

"""
    marginal_log_lhood(c::Component)

The marginal likelihood of data fit so far by `o`, which is the integral of the
likelihood given the mean and variance under the prior on those parameters.
"""
function marginal_log_lhood(c::Component{NormalInverseChisq{Float64}, NormalStats})
    μ0, σ20, κ0, ν0 = params(c.prior)
    μn, σ2n, κn, νn = params(posterior_canon(c.prior, c.suffstats))
    n = νn - ν0
    lgamma(νn*0.5) - lgamma(ν0*0.5) +
        0.5*(log(κ0)-log(κn)) +
        (0.5*ν0)*(log(ν0)+log(σ20)) - (0.5*νn)*(log(νn)+log(σ2n)) -
        (0.5*n)*log(π)
end
