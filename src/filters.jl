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
    1 .- pairwise(Hamming(), as') ./ size(as, 2)
end

"""
    randindex(ps::ParticleFilter, truth::Vector{<:Integer})

Compute the Rand Index of the clusters found by the particles in `ps`, given
ground truth cluster indices.  The Rand Index is the number of pairs of
observations on which the two clusterings agree, divided by the total number of
pairs.

"""
function randindex(ps::ParticleFilter, truth::Vector{<:Integer})
    ps_sim = assignment_similarity(ps)
    truth_sim = truth .== truth'

    N = length(ps_sim)

    both_same = sum(ps_sim .* truth_sim)
    # sum((1 .- ps_sim) .* (1 .- truth_sim)) = sum(1 - truth_sim - ps_sim + truth_sim*ps_sim)
    both_diff = N - sum(truth_sim) - sum(ps_sim) + both_same

    rand_index = (both_same + both_diff) / N
end
