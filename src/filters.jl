using ProgressMeter
using Distances

abstract type ParticleFilter end

function Base.filter!(ps::ParticleFilter, ys::AbstractVector{T} where T, progress=true;
                      cb=(ps,y)->nothing)
    @showprogress (progress ? 1 : Inf) "Fitting particles..." for y in ys
        fit!(ps, y)
        cb(ps, y)
    end
    ps
end

particles(p::ParticleFilter) = p.particles

Statistics.mean(f::Function, p::ParticleFilter) =
    mean(f.(particles(p)), Weights(weight.(particles(p))))

state_entropy(p::ParticleFilter) = mean(state_entropy, p)

normalize!(x) = (x ./= sum(x))
posterior_predictive(p::ParticleFilter) =
    MixtureModel(posterior_predictive.(particles(p)), normalize!(weight.(particles(p))))

function assignments(p::ParticleFilter)
    ps = particles(p)
    a = assignments(first(ps))
    asgn = Matrix{eltype(a)}(undef, length(a), length(ps))
    for (i,p) in enumerate(ps)
        asgn[:,i] .= assignments(p)
    end
    asgn
end

ncomponents_dist(p::ParticleFilter) =
    fit(Categorical, ncomponents.(p.particles), weight.(p.particles))

function assignment_similarity(ps::ParticleFilter)
    as = assignments(ps)
    1 .- pairwise(Hamming(), as, dims=1) ./ size(as, 2)
end

"""
    randindex(ps::ParticleFilter, truth::Vector{<:Integer})

Compute the Rand Index of the clusters found by the particles in `ps`, given
ground truth cluster indices.  The Rand Index is the number of pairs of
observations on which the two clusterings agree, divided by the total number of
pairs.

"""
function randindex(ps::ParticleFilter, truth::T, type=:adjusted) where {T<:AbstractVector{<:Integer}}
    ps_sim = assignment_similarity(ps)
    truth_sim = truth .== truth'

    size(truth_sim) == size(ps_sim) ||
        throw(DimensionMismatch("Ground truth and particles have different number of obs!"))

    # number of pairs 
    N = prod(size(ps_sim))

    # adjustment needs the sum of squares of the clusters sizes, which is just
    # the area of each assignment similarity matrix that's 1
    nis = sum(ps_sim)
    njs = sum(truth_sim)

    both_same = sum(ps_sim .* truth_sim)
    # sum((1 .- ps_sim) .* (1 .- truth_sim)) = sum(1 - truth_sim - ps_sim + truth_sim*ps_sim)
    both_diff = N - nis - njs + both_same

    if type == :adjusted
        n = length(truth)
        # from Clustering.jl:
        nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))
        adj_rand_index = (both_same + both_diff - nc) / (N - nc)
    else
        rand_index = (both_same + both_diff) / N
    end
end
