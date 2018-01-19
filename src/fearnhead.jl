using ProgressMeter

# some potential optimizations: 
# * make particles themselves immutable and avoid copying
# * keep track of assignments at population level, not particles (no need to
#   copy vectors on particle copy/propogation, just update stats)
# * use ancestor indices to link from one iteration to the next. (take advantage
#   of shared ancestry).  requires reconstructing the full assignment vectors
#   afterwards but who cares at that point.
#
# all of this means we can get rid of hte online stat thing...I don't know it's
# really approrpiate here anyway...


mutable struct FearnheadParticles
    particles::Vector{AbstractParticle}
    N::Int
end

function Base.show(io::IO, ps::FearnheadParticles)
    n = length(ps.particles)
    if n < ps.N
        println("Particle filter with $n (up to $(ps.N)) particles:")
    else
        println("Particle filter with $n particles:")
    end
    showcompact(io, ps.particles)
end

# Initialize population with a single, empty particle. (avoid redundancy)
FearnheadParticles(n::Int, priors...) = FearnheadParticles([Particle(priors...)], n)
FearnheadParticles(n::Int, prior::Union{Tuple,<:Distribution}, α::Float64) = FearnheadParticles([InfiniteParticle(prior, α)], n)


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
function fit!(ps::FearnheadParticles, y::Float64)
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
        ps.particles = typeof(ps.particles)(ps.N)

        # propagate particles 1:ci-1
        # ps.particles[1:ci-1] .= putative[1:ci-1]
        for i in 1:ci-1
            ps.particles[i] = weight(putative[i], weight(putative[i])/totalw)
        end

        # resample the rest:
        wsample!(view(putative, ci:M),        # from putative particles ci:M...
                 view(ws, ci:M),              # weighted according to old weights...
                 view(ps.particles, ci:ps.N), # draw ps.N-ci+1 particles and store
                                              # in ps.particles...
                 replace=false)               # without replacement
        # set weights
        c /= totalw
        for i in ci:ps.N
            ps.particles[i] = weight(ps.particles[i], c)
        end
        @debug "  total weight: $(sum(weight(p) for p in ps.particles))"
    end
    ps
end

# just ignore weights in fit!
fit!(ps::FearnheadParticles, y::Float64, w::Float64) = fit!(ps, y)

function fit!(ps::FearnheadParticles, ys::AbstractVector{Float64})
    @showprogress 1 "Fitting particles..." for y in ys
        fit!(ps, y)
    end
    ps
end


function normalize_clusters!(ps::FearnheadParticles, method::Symbol)
    if method == :sort
        sort!.(ps.particles)
    else
        throw(ArgumentError("Method $method not supported"))
    end
    return ps
end


posterior_predictive(ps::FearnheadParticles) = MixtureModel(posterior_predictive.(ps.particles), weight.(ps.particles))
