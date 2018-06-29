using ProgressMeter
using Distances

abstract type ParticleFilter end

function Base.filter!(ps::ParticleFilter, ys::AbstractVector{T} where T, progress=true; cb=(ps,y)->nothing)
    @showprogress (progress ? 1 : Inf) "Fitting particles..." for y in ys
        fit!(ps, y)
        cb(ps, y)
    end
    ps
end

particles(p::ParticleFilter) = p.particles

normalize!(x) = (x ./= sum(x))

posterior_predictive(p::ParticleFilter) =
    MixtureModel(posterior_predictive.(particles(p)), normalize!(weight.(particles(p))))

function assignments(p::ParticleFilter)
    ps = particles(p)
    a = assignments(first(ps))
    asgn = Matrix{eltype(a)}(length(a), length(ps))
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
