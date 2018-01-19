abstract type AbstractParticle end

# a particle for parametric clustering
struct Particle{P,S,A} <: AbstractParticle
    components::Vector{Component{P,S}}
    ancestor::A
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

weight(p::Particle) = p.weight
# now we have a problem: need to be able to change weight of particle...
# weight!(p::Particle, w::Float64) = 
