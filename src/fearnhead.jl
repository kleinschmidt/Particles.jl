mutable struct FearnheadParticles{P} <: ExactStat{0}
    particles::Vector{P}
    N::Int
end

# Initialize population with a single, empty particle. (avoid redundancy)
FearnheadParticles(n::Int, priors...) = FearnheadParticles([Particle(priors...)], n)
FearnheadParticles(n::Int, prior::Tuple, α::Float64) = FearnheadParticles([InfiniteParticle(prior, α)], n)


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

"""
    fit!(ps::FearnheadParticles, y::Float64)

Filter a single observation with the population of particles in `ps`.

"""
function fit!(ps::FearnheadParticles{P}, y::Float64) where P
    # generate putative particles
    putative = mapreduce(p->putatives(p,y), vcat, Particle[], ps.particles)
    total_w = sum(weight(p) for p in putative)

    M = length(putative)
    if M <= ps.N
        @debug "  M=$M: Fewer than N=$(ps.N) particles"
        ps.particles = putative
    else
        @debug "  M=$M: More than N=$(ps.N): Resampling"
        # resample down to N particles
        sort!(putative, by=weight, rev=true)
        ws = weight.(putative)
        ci, c, totalw = cutoff(ws, ps.N)
        @debug "  keeping $(ci-1) out of $M (cutoff=$c)"
        ps.particles = Vector{P}(ps.N)
        # propagate particles 1:ci-1
        ps.particles[1:ci-1] .= putative[1:ci-1]
        # resample the rest:
        wsample!(view(putative, ci:M),          # draw from putative particles ci:M
                 ws[ci:M],                      # weight according to old weights
                 view(ps.particles, ci:ps.N),   # draw ps.N-ci+1 particles and store in ps.particles
                 replace=false)                 # without replacement

        foreach(p->weight!(p, weight(p)/totalw), view(ps.particles, 1:ci-1))
        foreach(p->weight!(p, c/totalw), view(ps.particles, ci:ps.N))
        @debug "  total weight: $(sum(weight(p) for p in ps.particles))"
    end
    ps
end

# just ignore weights in fit!
fit!(ps::FearnheadParticles, y::Float64, w::Float64) = fit!(ps, y)
