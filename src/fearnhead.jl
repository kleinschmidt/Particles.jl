mutable struct FearnheadParticles <: ExactStat{0}
    particles::Vector{Particle}
    N::Int
end

# Initialize population with a single, empty particle. (avoid redundancy)
FearnheadParticles(n::Int, priors...) = FearnheadParticles([Particle(priors...)], n)

"""
    cutoff(ws::Vector{<:Real}, N::Int)

Find the cutoff for weights to automatically propogate.  Returns

* `i`, the index of the first weight _not_ automatically propogated
* `1/c`, the **unnormalized** weight for the resampled particles
* `tot0` the total weight (for normalization)
"""
function cutoff(ws::Vector{<:Real}, N::Int)
    tot0 = sum(ws)
    tot = tot0
    for i in eachindex(ws)
        if tot / ws[i] + (i-1) >= N
            return i, tot/(N-i+1), tot0
        end
        tot -= ws[i]
    end
end
