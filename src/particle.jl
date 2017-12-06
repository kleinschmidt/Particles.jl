## Fearnhead non-parametric clustering

abstract type AbstractParticle end

Base.copy(p::T) where T<:AbstractParticle = deepcopy(p)
weight(p::AbstractParticle) = p.weight
weight!(p::AbstractParticle, w::Real) = (p.weight=w; p)

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
