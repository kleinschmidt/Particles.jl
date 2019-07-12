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
FearnheadParticles(n::Int, prior::Union{Tuple,<:Distribution}, stateprior::T) where T<:StatePrior =
    FearnheadParticles([InfiniteParticle(prior, stateprior)], n)

"""
    cutoff_ascending(ws::Vector{<:Real}, N::Int)

Find the cutoff for weights to automatically propogate, assuming that ws are
sorted in ascending order.  Returns

* `i`, the index of the first weight that is kept
* `1/c`, the **unnormalized** weight for the resampled particles

# Algorithm from Fearnhead and Clifford (2003):

We want to find `c` s.t. `sum(min(w/c, 1) for w in ws) = N`.  We do this by
finding the minimum element of `ws` κ such that `sum(min(w/κ, 1) for w in ws) ≤
N`.

The intuition is that for all w > κ you get 1, so if κ < all w, then you get M >
N.  as you move κ up the ws, you take away some of the 1s.  Let B_κ be the sum
of all w < κ, and A_κ be the number of elements ≥ κ.  Then we need B_κ / κ + A_κ
≤ N.

Once we have κ=ws[i], we're going to keep everything κ and higher (ws[i:end]).
which means that c ≤ κ.  so we have B_κ / c + A_κ = N => 1/c = (N-A_κ) /
(B_κ-κ).

The returned index is the index of the first w that's **kept**.  the returned w is
the weight that's assigned to the resampled particles.
"""
function cutoff_ascending(ws::Vector{T}, N::Int) where {T<:Real}
    issorted(ws) ||
        throw(ArgumentError("weights must be sorted in ascending order"))
    tot = zero(T)
    M = length(ws)
    for (i,w) in enumerate(ws)
        # avoid comparison with zero (will stop early)
        if !iszero(w)
            # there are i-1 elements < w, so M-(i-1) that are ≥ w
            n_geq = M-i+1
            if tot ≤ (N - n_geq)*w
                return i, tot / (N-n_geq)
            end
            tot += w
        end
    end
    # resample all, return M+1 and 1/N
    return M+1, tot/N
end

"""
    sample_stratified!(x::AbstractVector, n::Int, w::AbstractVector)

Stratified sampling algorithm from Carpenter et al. (1999), as described in
Fearnhead and Clifford (2003).  Draws a sample from `x` with weights `w`,
without replacement.

NOTE!! This algorithm is only guaranteed to give at most `n` samples, and not
necessarily exactly `n`, especially when `n` is close to `length(x)`.  This
isn't a problem in practice because this is used to sample from the "leftover"
putative particles, and the worst case is when there's no particles carried over
in which case we'll take at most 50% of the putatives (much less when there are
more than two components per particle).

Returns the number or elements resampled (≤ n)

Modifies the input vector, placing the resampled elements in the first positions
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
    i_store < n && @debug "sample_stratified! ask for $(n) but got $(i_store)"
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
        ps.particles = instantiate.(putative, weight.(putative) ./ total_w)
    else
        # TODO: consider doing this all in place, see Fearnhead and Cliffor
        # 2003, Appendix C.  basic idea is to use an algorithm like
        # median-finding (quickselect) to find the cutoff weight and partition
        # the putative particles into kept and re-sampled (in place), and then
        # re-sample the rest in place as well.  then you just trim the vector of
        # putatives to the right length and good to go.

        # resample down to N particles
        sort!(putative, alg=QuickSort, by=weight)
        ws = weight.(putative)
        # will keep kept_i:end, and resample from 1:(kept_i-1) and give weight w_resamp
        kept_i, w_resamp = cutoff_ascending(ws, ps.N)

        # keep kept_i:end
        n_kept = length(ws) - (kept_i - 1)
        
        # resample from 1:(kept_i-1)
        n_resamp = ps.N - n_kept

        @debug """
               M=$M: More than N=$(ps.N)
                 Keeping $n_kept ($kept_i:end)
                 Resampling $(M-n_kept) from $n_resamp
               """

        n_resamp = sample_stratified!(view(putative, 1:(kept_i-1)),
                                      n_resamp,
                                      view(ws, 1:(kept_i-1)))

        resize!(ps.particles, n_resamp+n_kept)

        @debug "resampling $n_resamp to ps.particles[$(1:n_resamp)]"
        @views ps.particles[1:n_resamp] .=
            instantiate.(putative[1:n_resamp], w_resamp/total_w)

        @debug "keeping last $n_kept ($(kept_i+1:lastindex(putative))) " *
            "as ps.particles[$(n_resamp .+ (1:n_kept))]"
        @views ps.particles[n_resamp .+ (1:n_kept)] .=
            instantiate.(putative[kept_i:end],
                         weight.(putative[kept_i:end]) ./ total_w)

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
