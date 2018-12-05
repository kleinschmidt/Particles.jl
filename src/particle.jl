struct PutativeParticle{P,O,S,C}
    ancestor::P
    obs::O
    state::S
    updated_comp::C
    weight::Float64
end

instantiate(p::PutativeParticle{P}) where P =
    error("Don't know how to instantiate a putative $P")

weight(p::PutativeParticle) = p.weight


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

putatives(p::InfiniteParticle, y) =
    (PutativeParticle(p, y, j) for j in candidates(p.stateprior))

# lazily compute the weight without actually updating the suff stats etc, which
# requries expensive copying
function PutativeParticle(p::InfiniteParticle, obs::O, state::S) where {O,S}
    logweight = log(p.weight)
    logweight += log_prior(p.stateprior, state)
    comp_i = state_to_index(p.stateprior, state)
    if comp_i ≤ length(p.components)
        # likelihood adjustment for old observations
        comp = p.components[comp_i]
        logweight -= marginal_log_lhood(comp)
    else
        comp = p.prior
    end
    updated_comp = add(comp, obs)
    logweight += marginal_log_lhood(add(comp, obs))
    
    return PutativeParticle(p, obs, state, updated_comp, exp(logweight))
    
end

function instantiate(putative::PutativeParticle{<:InfiniteParticle})
    p = putative.ancestor
    stateprior, comp_i = add(p.stateprior, putative.state)
    0 < comp_i ≤ length(p.components)+1 ||
        throw(ArgumentError("can't fit component $comp_i: must be between 0 and " *
                            "$(length(p.components)+1)"))

    components = copy(p.components)
    if comp_i > length(components)
        push!(components, putative.updated_comp)
    else
        components[comp_i] = putative.updated_comp
    end

    return InfiniteParticle(components, p, comp_i, putative.weight, p.prior, stateprior)
end

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
    0 < x ≤ length(p.components)+1 ||
        throw(ArgumentError("can't fit component $x: must be between 0 and " *
                            "$(length(p.components)+1)"))

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
Distributions.ncomponents(p::InfiniteParticle, includeprior::Bool=false) =
    length(p.components) + includeprior

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
    marginal_posterior(p::AbstractParticle)

The (unnormalized) posterior probability of the parameters in `p` given the data
`fit!` by it thus far.
"""
marginal_posterior(p::AbstractParticle) = exp(marginal_log_posterior(p))

marginal_log_posterior(p::InfiniteParticle) =
    sum(marginal_log_lhood(c) for c in components(p)) + marginal_log_prior(p.stateprior)

# TODO: normalize clusters (sort based on mean, and re-write assignments, or
# provide a view...
