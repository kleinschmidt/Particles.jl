using ProgressMeter

abstract type ParticleFilter end

function fit!(ps::ParticleFilter, ys::AbstractVector{Float64}, progress=true)
    if progress
        @showprogress 1 "Fitting particles..." for y in ys
            fit!(ps, y)
        end
    else
        for y in ys
            fit!(ps, y)
        end
    end
    ps
end

particles(p::ParticleFilter) = p.particles

posterior_predictive(p::ParticleFilter) =
    MixtureModel(posterior_predictive.(particles(p)), weight.(particles(p)))

function assignments(p::ParticleFilter)
    ps = particles(p)
    a = assignments(first(ps))
    asgn = Matrix{eltype(a)}(length(a), length(particles))
    for (i,p) in ps
        asgn[:,i] .= assignments(p)
    end
    asgn
end

function ncomponents_dist(p::ParticleFilter)
    ps = particles(p)
    ncomps = ncomponents.(ps)
    weights = weight.(ps)
    comp_ps = zeros(maximum(ncomps))
    for (n,w) in zip(ncomps, weights)
        comp_ps[n] += w
    end
    Categorical(comp_ps ./ sum(comp_ps))
end
