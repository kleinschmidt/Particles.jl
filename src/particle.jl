## Fearnhead non-parametric clustering

abstract type AbstractParticle end

Base.copy(p::T) where T<:AbstractParticle = deepcopy(p)
weight(p::AbstractParticle) = p.weight
weight!(p::AbstractParticle, w::Real) = (p.weight=w; p)

function Base.sort!(p::T) where T<:AbstractParticle
    ix = sortperm(p.components, by=value)
    p.components .= p.components[ix]
    # reverse mapping: rev_ix[i] is the new index for old index i
    rev_ix = sortperm(ix)
    p.assignments .= rev_ix[p.assignments]
    return p
end


"""
A Particle holds the `FitNormalInverseChisq` for each category, the category
assignments ``x`` the weight for the particle.
"""
mutable struct Particle{T} <: AbstractParticle
    components::Vector{T}
    assignments::Vector{Int}
    weight::Float64
end

Particle(priors...) = Particle([FitNormalInverseChisq(p) for p in priors], Int[], 1.0)
putatives(p::Particle, y::Real) = [fit!(copy(p), y, j) for j in 1:length(p.components)]

function fit!(p::Particle, y::Float64, x::Int)
    old_log_lhood = marginal_log_lhood(p.components[x])
    fit!(p.components[x], y)
    p.weight = p.weight * exp(marginal_log_lhood(p.components[x]) - old_log_lhood)
    push!(p.assignments, x)
    return p
end


"""
An InfiniteParticle holds a potentially infinite number of components, 
potentially expanding every time it generates putatives.
"""
mutable struct InfiniteParticle{T} <: AbstractParticle
    components::Vector{T}
    assignments::Vector{Int}
    weight::Float64
    prior::T
    α::Float64                  # prior count for new cluster
end

InfiniteParticle(prior::T, α::Float64) where T = InfiniteParticle{T}(T[], Int[], 1.0, prior, α)

InfiniteParticle(params::NTuple{4,Float64}, α::Float64) =
    InfiniteParticle(FitNormalInverseChisq(params), α)

putatives(p::InfiniteParticle, y::Real) =
    [fit!(copy(p), y, j) for j in 1:length(p.components)+1]

"""
    fit!(p::InfiniteParticle, y::Real, x::Int)

Update `p`, classifying `y` under cluster `x`.  `x` can be an existing cluster
or a new one.  The corresponding component is updated, the classification
recorded, and the weight is updated.  Terms that are constant across all other
particles which have seen the same data are not included in the weight update.

The weight update term is the ratio of the previous and updated marginal
posterior:
\[
\frac{p((z_{1:n}, x) | y_{1:n+1} )}{p(z_{1:n} | y_{1:n} )}
\propto
\frac{p(y_{1:n+1} | (z_{1:n}, x)) p((z_{1:n}, x))}{p(y_{1:n} | z_{1:n}) p(z_{1:n})}
\]

The prior ratio can be reduced using the conditional distribution: $p(z_{1:n},
x)/p(z_{1:n}) = p(x | z_{1:n})$.  Under a Chinese Restaurant Process prior, this
is $N_x / \sum_j N_j + \alpha$ if $x$ corresponds to an existing component and
$\alpha / \sum_j N_j + \alpha$ if it's new.  Because the total count $\sum_j
N_j$ is the same across particles, the net change is proportional to $N_x$ or
$\alpha$.

The likelihood ratio also depends on whether $y$ is being assigned to a new
component or not.  If it is being assigned to a new component, then its marginal
likelihood is independent of all other points and the entire adjustment is
proportional to the marginal likelihood of $y$ under the prior.  When $x$
corresponds to an existing component, the adjustment is proportional to the
ratio of the marginal likelihood of that component before and after
incorporating $y$: $\frac{p(y_{x_i=x}, y_{n+1})}{p(y_{x_i=x})}$.

"""
function fit!(p::InfiniteParticle, y::Real, x::Int)
    @argcheck 0 < x ≤ length(p.components)+1
    Δlogweight = 0
    if x ≤ length(p.components)
        # likelihood adjustment for old observations
        Δlogweight -= marginal_log_lhood(p.components[x])
        # prior ∝ N_i
        Δlogweight += log(nobs(p.components[x]))
    else
        push!(p.components, copy(p.prior))
        # prior ∝ α
        Δlogweight += log(p.α)
    end
    fit!(p.components[x], y)
    # likelihood of new observation
    Δlogweight += marginal_log_lhood(p.components[x])
    p.weight = exp(log(p.weight) + Δlogweight)
    push!(p.assignments, x)
    return p
end
