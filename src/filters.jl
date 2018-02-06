using ProgressMeter

abstract type ParticleFilter end

function fit!(ps::ParticleFilter, ys::AbstractVector{Float64}, progress=true)
    @showprogress (progress ? 1 : Inf) "Fitting particles..." for y in ys
        fit!(ps, y)
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
