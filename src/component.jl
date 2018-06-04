struct Component{P,S}
    prior::P
    suffstats::S
end

Component(prior::Distribution) = Component(prior, empty_suffstats(prior))

Component(params::NTuple{N,<:Real}) where N = Component(NormalInverseChisq(params...))

Base.show(io::IO, c::Component{P,S}) where {P,S} = print(io, "$P$(params(c)) w/ n=$(nobs(c))")

posterior_predictive(gc::Component) =
    posterior_predictive(posterior_canon(gc.prior, gc.suffstats))

posterior_predictive(d::NormalInverseChisq) =
    LocationScale(d.μ, sqrt((1+d.κ)*d.σ2/d.κ), TDist(d.ν))

function posterior_predictive(d::NormalInverseWishart)
    df = d.nu - length(d.mu) + 1
    Λ = d.Lamchol[:U]'*d.Lamchol[:U]
    MvTDist(df, d.mu, Λ*(d.kappa+1)/(d.kappa*df))
end

Distributions.logpdf(c::Component, x) = logpdf(posterior_predictive(c), x)
Distributions.pdf(c::Component, x) = logpdf(posterior_predictive(c), x)

Distributions.params(c::Component) = params(posterior_canon(c.prior, c.suffstats))

empty_suffstats(d::NormalInverseChisq) = NormalStats(0, 0, 0, 0)
function empty_suffstats(d::NormalInverseWishart)
    d = d.dim
    MvNormalStats(zeros(d), zeros(d), zeros(d,d), 0.)
end

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

function add(ss::MvNormalStats, x::AbstractVector)
    tw = ss.tw + 1
    Δ = x - ss.m
    m = ss.m + Δ/tw
    s2 = tw>1 ? ss.s2 + Δ*(x.-m)' : zeros(size(ss.s2))
    return MvNormalStats(ss.s+x, m, s2, tw)
end

function sub(ss::MvNormalStats, x::AbstractVector)
    tw = ss.tw-1
    if tw ≤ 0
        n = length(ss.m)
        return MvNormalStats(zeros(n), zeros(n), zeros(n,n), 0)
    else
        Δ = x - ss.m
        m = ss.m - Δ./tw
        s2 = ss.s2 - Δ*(x-m)'
        return MvNormalStats(ss.s-x, m, s2, tw)
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
marginal_lhood(c::Component) =
    marginal_lhood(c.prior, c.suffstats)

marginal_lhood(prior, suffstats) = exp(marginal_log_lhood(prior, suffstats))

function marginal_lhood(prior::NormalInverseChisq{Float64}, suffstats::NormalStats)
    μ0, σ20, κ0, ν0 = params(prior)
    μn, σ2n, κn, νn = params(posterior_canon(prior, suffstats))
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
marginal_log_lhood(c::Component) =
    marginal_log_lhood(c.prior, c.suffstats)

function marginal_log_lhood(prior::NormalInverseChisq{Float64}, suffstats::NormalStats)
    μ0, σ20, κ0, ν0 = params(prior)
    μn, σ2n, κn, νn = params(posterior_canon(prior, suffstats))
    n = νn - ν0
    lgamma(νn*0.5) - lgamma(ν0*0.5) +
        0.5*(log(κ0)-log(κn)) +
        (0.5*ν0)*(log(ν0)+log(σ20)) - (0.5*νn)*(log(νn)+log(σ2n)) -
        (0.5*n)*log(π)
end

function marginal_log_lhood(prior::NormalInverseWishart, suffstats::MvNormalStats)
    μ0, Λchol0, κ0, ν0 = params(prior)
    μn, Λcholn, κn, νn = params(posterior_canon(prior, suffstats))
    n = νn - ν0
    d = length(μ0)
    
    (-0.5*n*d) * logπ +
        logmvgamma(d, νn*0.5) - logmvgamma(d, ν0*0.5) +
        logdet(Λchol0)*ν0*0.5 - logdet(Λcholn)*νn*0.5 +
        d*0.5*(log(κ0) - log(κn))
end
