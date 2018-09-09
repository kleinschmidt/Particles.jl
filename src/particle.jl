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
    ancestor::Union{Nothing,Particle}
    assignment::Int
    weight::Float64
end

function Base.show(io::IO, p::Particle{P,S}) where {P,S}
    if get(io, :compact, false)
        print(io, "$(length(p.components))-Particle")
    else
        println(io, "Particle with $(length(p.components)) components:")
        for c in p.components
            println(io, "  $c")
        end
    end
end

Particle(priors::AbstractVector{D}) where D<:Distribution = 
    Particle(Component.(priors), nothing, 0, 1.0)
Particle(priors::D...) where D<:Distribution =
    Particle([Component.(priors)...], nothing, 0, 1.0)
Particle(params::NTuple{4,<:Real}...) = Particle([Component.(params)...], nothing, 0, 1.0)

function fit(p::Particle, y, x::Int)
    comps = copy(p.components)
    old_llhood = marginal_log_lhood(comps[x])
    comps[x] = add(comps[x], y)
    new_llhood = marginal_log_lhood(comps[x])
    Particle(comps, p, x, p.weight * exp(new_llhood - old_llhood))
end

putatives(p::Particle, y) = (fit(p, y, x) for x in eachindex(p.components))

weight(p::Particle, w::Float64) = Particle(p.components, p.ancestor, p.assignment, w)

Distributions.ncomponents(p::Particle) = length(p.components)

"""
An InfiniteParticle holds a potentially infinite number of components,
potentially expanding every time it generates putatives.
"""
mutable struct InfiniteParticle{P,S,T} <: AbstractParticle
    components::Vector{Component{P,S}}
    ancestor::Union{Nothing,InfiniteParticle}
    assignment::Int
    weight::Float64
    prior::Component{P,S}
    stateprior::T
end

function Base.show(io::IO, p::InfiniteParticle)
    if get(io, :compact, false)
        print(io, "$(length(p.components))+ Particle")
    else
        println("Particle with $(length(p.components))+ components:")
        for c in p.components
            println(io, "  $c")
        end
        println(io, "  (prior: $(p.prior))")
    end
end

InfiniteParticle(prior::Component, α::Float64) =
    InfiniteParticle(prior, ChineseRestaurantProcess(α))
InfiniteParticle(prior::Component, stateprior::T) where T<:StatePrior =
    InfiniteParticle(typeof(prior)[], nothing, 0, 1.0, prior, stateprior)
InfiniteParticle(prior, stateprior) = InfiniteParticle(Component(prior), stateprior)

weight(p::InfiniteParticle, w::Float64) =
    InfiniteParticle(p.components, p.ancestor, p.assignment, w, p.prior, p.stateprior)

putatives(p::InfiniteParticle, y) = (fit(p, y, j) for j in candidates(p.stateprior))

"""
    fit(p::InfiniteParticle, y::Real, x::Int)

Update `p`, classifying `y` under cluster `x`.  `x` can be an existing cluster
or a new one.  The corresponding component is updated, the classification
recorded, and the weight is updated.  Terms that are constant across all other
particles which have seen the same data are not included in the weight update.

The weight update term is the ratio of the previous and updated marginal
posterior:
```math
\\frac{p((z_{1:n}, x) | y_{1:n+1} )}{p(z_{1:n} | y_{1:n} )} ∝
\\frac{p(y_{1:n+1} | (z_{1:n}, x)) p((z_{1:n}, x))}{p(y_{1:n} | z_{1:n}) p(z_{1:n})}
```

The prior ratio can be reduced using the conditional distribution: ``p(z_{1:n},
x)/p(z_{1:n}) = p(x | z_{1:n})``.  Under a Chinese Restaurant Process prior, this
is ``N_x / ∑_j N_j + α`` if ``x`` corresponds to an existing component and
``α / ∑_j N_j + α`` if it's new.  Because the total count ``∑_j
N_j`` is the same across particles, the net change is proportional to ``N_x`` or
``α``.

The likelihood ratio also depends on whether ``y`` is being assigned to a new
component or not.  If it is being assigned to a new component, then its marginal
likelihood is independent of all other points and the entire adjustment is
proportional to the marginal likelihood of ``y`` under the prior.  When ``x``
corresponds to an existing component, the adjustment is proportional to the
ratio of the marginal likelihood of that component before and after
incorporating ``y``: ``\\frac{p(y_{x_i=x}, y_{n+1})}{p(y_{x_i=x})}``.

"""
function fit(p::InfiniteParticle, y, x::Int)
    # first calculate log-prior
    Δlogweight = log_prior(p.stateprior, x)
    # then update sufficient stats and convert x to an index
    stateprior, x = add(p.stateprior, x)
    @argcheck 0 < x ≤ length(p.components)+1

    components = copy(p.components)
    if x ≤ length(components)
        # likelihood adjustment for old observations
        Δlogweight -= marginal_log_lhood(components[x])
    else
        push!(components, p.prior)
    end
    components[x] = add(components[x], y)
    # likelihood of new observation
    Δlogweight += marginal_log_lhood(components[x])
    weight = exp(log(p.weight) + Δlogweight)

    return InfiniteParticle(components, p, x, weight, p.prior, stateprior)
end



Distributions.components(p::InfiniteParticle) = [p.components..., p.prior]
Distributions.ncomponents(p::InfiniteParticle, includeprior::Bool=false) = length(p.components) + includeprior

weights(p::Particle) = ones(length(p.components)) ./ length(p.components)
weights(p::InfiniteParticle) = exp.(log_prior(p.stateprior))

state_entropy(p::InfiniteParticle) = entropy(p.stateprior)

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
marginal_log_posterior(p::InfiniteParticle) =
    sum(marginal_log_lhood(c) for c in components(p)) + marginal_log_prior(p.stateprior)

# TODO: normalize clusters (sort based on mean, and re-write assignments, or
# provide a view...
