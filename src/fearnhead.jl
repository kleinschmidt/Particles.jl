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


mutable struct FearnheadParticles{P} <: ParticleFilter
    particles::Vector{P}
    N::Int
end

function Base.show(io::IO, ps::FearnheadParticles)
    n = length(ps.particles)
    if n < ps.N
        println(io, "Particle filter with $n (up to $(ps.N)) particles:")
    else
        println(io, "Particle filter with $n particles:")
    end
    show(IOContext(io, :compact=>true), ps.particles)
end

# Initialize population with a single, empty particle. (avoid redundancy)
FearnheadParticles(n::Int, priors...) = FearnheadParticles([Particle(priors...)], n)
FearnheadParticles(n::Int, prior::Union{Tuple,<:Distribution}, stateprior::T) where T<:StatePrior =
    FearnheadParticles([InfiniteParticle(prior, stateprior)], n)


"""
    cutoff(ws::Vector{<:Real}, N::Int)

Find the cutoff for weights to automatically propogate.  Returns

* `i`, the index of the first weight _not_ automatically propogated
* `1/c`, the **unnormalized** weight for the resampled particles
* `tot0` the total weight (for normalization)
"""
function cutoff(ws::Vector{<:Real}, N::Int)
    # TODO: do this in place (see Fearnhead and Clifford 2003, Appendix C)
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
    sample_stratified!(x::AbstractVector, n::Int, w::AbstractVector)

Stratified sampling algorithm from Carpenter et al. (1999), as described in
Fearnhead and Clifford (2003).  Draws a sample from `x` with weights `w`,
without replacement.

NOTE!! This algorithm is only guaranteed to give at most `n` samples, and not
necessarily exactly `n`, especially when `n` is close to `length(x)`.  This
isn't a problem in practice because this used to sample from the "leftover"
putative particles, and the worst case is when there's no particles carried over
in which case we'll take at most 50% of the putatives (much less when there are
more than two components per particle).

Returns the number or elements resampled (≤ n)

"""
function sample_stratified!(x::AbstractVector, n::Int, w)
    K = sum(w) / n
    U = rand() * K

    i_store = 0
    for i in eachindex(x,w)
        U = U-w[i]
        if U < 0
            i_store += 1
            x[i_store] = x[i]
            U += K
        end
    end

    @assert i_store ≤ n
    i_store < n && @debug "  sample_stratified! ask for $(n) but got $(i_store)"
    return i_store
end

function sample_stratified(x::AbstractVector, n, w)
    x_samp = copy(x)
    n_samp = sample_stratified!(x_samp, n, w)
    resize!(x_samp, n_samp)
    return x_samp
end


"""
    fit!(ps::FearnheadParticles, y::Float64)

Filter a single observation with the population of particles in `ps`.

"""
function fit!(ps::FearnheadParticles{P}, y) where P
    # generate putative particles
    putative = collect(Iterators.flatten(putatives(p, y) for p in ps.particles))
    total_w = sum(weight(p) for p in putative)

    M = length(putative)
    if M <= ps.N
        @debug "  M=$M: Fewer than N=$(ps.N) particles"
        ps.particles = instantiate.(putative)
    else
        # TODO: consider doing this all in place, see Fearnhead and Cliffor
        # 2003, Appendix C.  basic idea is to use an algorithm like
        # median-finding (quickselect) to find the cutoff weight and partition
        # the putative particles into kept and re-sampled (in place), and then
        # re-sample the rest in place as well.  then you just trim the vector of
        # putatives to the right length and good to go.

        @debug "  M=$M: More than N=$(ps.N): Resampling"
        # resample down to N particles
        sort!(putative, alg=QuickSort, by=weight, rev=true)
        ws = weight.(putative)
        ci, c, totalw = cutoff(ws, ps.N)
        @debug "  keeping $(ci-1) out of $M (cutoff=$c)"
        
        # resample from ci:end
        n_resamp = ps.N - (ci-1)
        n_resamp = sample_stratified!(view(putative, ci:lastindex(putative)),
                                      n_resamp,
                                      view(ws, ci:lastindex(ws)))

        resize!(ps.particles, n_resamp+ci-1)

        for i in eachindex(ps.particles)
            new_w = (i < ci ? weight(putative[i]) : c) / total_w
            ps.particles[i] = weight(instantiate(putative[i]), new_w)
        end
        @debug "  total weight: $(sum(weight(p) for p in ps.particles))"
    end
    return ps
end

function normalize_clusters!(ps::FearnheadParticles, method::Symbol)
    if method == :sort
        sort!.(ps.particles)
    else
        throw(ArgumentError("Method $method not supported"))
    end
    return ps
end
