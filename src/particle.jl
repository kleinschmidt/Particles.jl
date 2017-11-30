## Fearnhead non-parametric clustering

"""
A Particle holds the `FitNormalInverseChisq` for each category, the category
assignments ``x`` the weight for the particle.
"""
mutable struct Particle{T}
    components::Vector{T}
    assignments::Vector{Int}
    weight::Float64
end

Particle(priors...) = Particle([FitNormalInverseChisq(p) for p in priors], Int[], 1.0)

Base.copy(p::Particle) = deepcopy(p)

function fit!(p::Particle, y::Float64, x::Int)
    old_log_lhood = marginal_log_lhood(p.components[x])
    fit!(p.components[x], y)
    p.weight = p.weight * exp(marginal_log_lhood(p.components[x]) - old_log_lhood)
    push!(p.assignments, x)
    p
end

weight(p::Particle) = p.weight
weight!(p::Particle, w::Real) = (p.weight=w; p)

putatives(p::Particle, y::Real) = [fit!(copy(p), y, j) for j in 1:length(p.components)]
