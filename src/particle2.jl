abstract type AbstractParticle end

weight(p::AbstractParticle) = p.weight
Distributions.components(p::AbstractParticle) = p.components


nobs(p::AbstractParticle) = sum(nobs(c) for c in components(p))
Base.isempty(p::AbstractParticle) = all(nobs(c) == 0 for c in components(p))

function assignments(p::AbstractParticle)
    asgns = Int[]
    while !isempty(p)
        push!(asgns, p.assignment)
        p = p.ancestor
    end
    reverse!(asgns)
end


# a particle for parametric clustering
struct Particle{P,S} <: AbstractParticle
    components::Vector{Component{P,S}}
    ancestor::Union{Void,Particle}
    assignment::Int
    weight::Float64
end

function Base.show(io::IO, p::Particle{P,S}) where {P,S}
    if get(io, :compact, false)
        showcompact(io, p)
    else
        println(io, "Particle with $(length(p.components)) components:")
        for c in p.components
            println(io, "  $c")
        end
    end
end

Base.showcompact(io::IO, p::Particle) = print(io, "$(length(p.components))-Particle")

Particle(params::NTuple{4,<:Real}...) = Particle([Component.(params)...], nothing, 0, 1.0)

function fit(p::Particle, y::Float64, x::Int)
    comps = copy(p.components)
    old_llhood = marginal_log_lhood(comps[x])
    comps[x] = add(comps[x], y)
    new_llhood = marginal_log_lhood(comps[x])
    Particle(comps, p, x, p.weight * exp(new_llhood - old_llhood))
end

putatives(p::Particle, y::Float64) = [fit(p, y, x) for x in eachindex(p.components)]

weight(p::Particle, w::Float64) = Particle(p.components, p.ancestor, p.assignment, w)
# now we have a problem: need to be able to change weight of particle...
# weight!(p::Particle, w::Float64) =


"""
An InfiniteParticle holds a potentially infinite number of components,
potentially expanding every time it generates putatives.
"""
mutable struct InfiniteParticle{P,S} <: AbstractParticle
    components::Vector{Component{P,S}}
    ancestor::Union{Void,InfiniteParticle}
    assignment::Int
    weight::Float64
    prior::Component{P,S}
    α::Float64                  # prior count for new cluster
end

function Base.show(io::IO, p::InfiniteParticle)
    if get(io, :compact, false)
        showcompact(io, p)
    else
        println("Particle with $(length(p.components))+ components:")
        for c in p.components
            println(io, "  $c")
        end
        println(io, "  (prior: $(p.prior))")
    end
end

Base.showcompact(io::IO, p::InfiniteParticle) =
    print(io, "$(length(p.components))+ Particle")

InfiniteParticle(prior::Component, α::Float64) =
    InfiniteParticle(typeof(prior)[], nothing, 0, 1.0, prior, α)
InfiniteParticle(prior, α::Float64) = InfiniteParticle(Component(prior), α)

weight(p::InfiniteParticle, w::Float64) =
    InfiniteParticle(p.components, p.ancestor, p.assignment, w, p.prior, p.α)

putatives(p::InfiniteParticle, y::Real) = [fit(p, y, j) for j in 1:length(p.components)+1]

"""
    fit(p::InfiniteParticle, y::Real, x::Int)

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

The prior ratio can be reduced using the conditional distribution: ``p(z_{1:n},
x)/p(z_{1:n}) = p(x | z_{1:n})``.  Under a Chinese Restaurant Process prior, this
is ``N_x / \sum_j N_j + \alpha`` if ``x`` corresponds to an existing component and
``\alpha / \sum_j N_j + \alpha`` if it's new.  Because the total count ``\sum_j
N_j`` is the same across particles, the net change is proportional to ``N_x`` or
``\alpha``.

The likelihood ratio also depends on whether ``y`` is being assigned to a new
component or not.  If it is being assigned to a new component, then its marginal
likelihood is independent of all other points and the entire adjustment is
proportional to the marginal likelihood of ``y`` under the prior.  When ``x``
corresponds to an existing component, the adjustment is proportional to the
ratio of the marginal likelihood of that component before and after
incorporating ``y``: ``\frac{p(y_{x_i=x}, y_{n+1})}{p(y_{x_i=x})}``.

"""
function fit(p::InfiniteParticle, y::Real, x::Int)
    @argcheck 0 < x ≤ length(p.components)+1
    Δlogweight = 0
    components = copy(p.components)
    if x ≤ length(components)
        # likelihood adjustment for old observations
        Δlogweight -= marginal_log_lhood(components[x])
        # prior ∝ N_i
        Δlogweight += log(nobs(components[x]))
    else
        push!(components, Component(p.prior))
        # prior ∝ α
        Δlogweight += log(p.α)
    end
    components[x] = add(components[x], y)
    # likelihood of new observation
    Δlogweight += marginal_log_lhood(components[x])
    weight = exp(log(p.weight) + Δlogweight)

    return InfiniteParticle(components, p, x, weight, p.prior, p.α)
end



Distributions.components(p::InfiniteParticle) = [p.components..., p.prior]

weights(p::Particle) = ones(length(p.components)) ./ length(p.components)
weights(p::InfiniteParticle) = (w = [Float64.(nobs.(p.components))..., p.α]; w ./= sum(w); w)
#weights(p::InfiniteParticle) = (w = Float64.(push!(nobs.(p.components), p.α)); w ./= sum(w); w)

"""
    posterior_predictive(p::P) where P<:AbstractParticle

Get the posterior predictive distribution for a particle, which is a mixture of
the posterior predictives for each component (including the prior, for an
`InfiniteParticle`).
"""

posterior_predictive(p::P) where P<:AbstractParticle =
    MixtureModel(posterior_predictive.(components(p)), weights(p))

"""
    marginal_posterior(p::Particle)

The (unnormalized) posterior probability of the parameters in `p` given the data
`fit!` by it thus far.
"""

marginal_posterior(p::AbstractParticle) = exp(marginal_log_posterior(p))

marginal_log_posterior(p::Particle) = sum(marginal_log_lhood(c) for c in components(p))
function marginal_log_posterior(p::InfiniteParticle)
    # prior is prod_i(α × (n_i-1)!) for each component i (since the prior is α
    # for the first obs in a new cluster and n_i thereafter.
    log_prior =
        length(p.components) * log(p.α) +
        sum(lgamma(nobs(c)) for c in p.components)
    log_lhood = sum(marginal_log_lhood(c) for c in p.components)
    return log_prior + log_lhood
end

