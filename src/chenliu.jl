# An alternative resampling algorithm described in Fearnhead (2004):
#
# for each particle, generate putatives (z_1:n^i, j)
# sample one successor proportional to p((z_1:n^i, j) | x_1:n+1)
# assign weight based on sum_1^k_i+1(p((z_1:n^i, j) | x_1:n+1)/p(z_1:n^i | x_1:n))

mutable struct ChenLiuParticles{P} <: ParticleFilter
    particles::Vector{P}
    N::Int
    rejuvination_threshold::Float64
end

function Base.show(io::IO, ps::ChenLiuParticles)
    n = length(ps.particles)
    if n < ps.N
        println("Particle filter with $n (up to $(ps.N)) particles:")
    else
        println("Particle filter with $n particles:")
    end
    showcompact(io, ps.particles)
end

# initialize with full population of empty particles, because this method
# doens't benefit from redundancy in the same way as the Fearnhead method does
ChenLiuParticles(n::Int, prior::Union{Tuple,<:Distribution}, α::Float64) =
    ChenLiuParticles([InfiniteParticle(prior, α) for _ in 1:n], n)

function propogate_chenliu(p::InfiniteParticle, y::Float64)
    ps = putatives(p, y)
    # sample next based on updated weights (which are proportional to the
    # posterior p( (z_1:n, j) | x_1:n+1 ) because they've been updated based on
    # the same ancestor, multiplying by
    # p( (z_1:n,j) | x_1:n+1) / p(z_1:n | x_1:n) ∝ p( (z_1:n,j) | x_1:n+1 )
    next = wsample(ps, weight.(ps))
    # now to update the weight.  we need it to be
    # w_n+1 = w_n × sum( p(z_1:n,j | x_1:n+1) / p(z_1:n | x_1:n) )
    # call the putative weight of particle j v_j.
    # v_j = w_n × p((z_1:n,j) | x_1:n+1) / p(z_1:n | x_1:n), so we just need to
    # add those up
    return weight(next, sum(weight(pp) for pp in ps))
end

"""
    coefvar(x)

Calculat the coefficient of variation for x: σ / μ * 100
"""
function coefvar(x)
    # use NormalStats to calculate both mean and var in one pass
    ss = suffstats(Normal, x)
    sqrt(ss.s2 / ss.tw) / ss.m * 100
end

function fit!(ps::ChenLiuParticles, y::Float64)
    broadcast!(propogate_chenliu, ps.particles, ps.particles, y)
    # rejuvinate if necessary
    ws = weight.(ps.particles)
    cv = coefvar(ws)
    if cv > ps.rejuvination_threshold
        @debug "  Rejuvinating ($cv > $(ps.rejuvination_threhold))"
        ps.particles = wsample(ps.particles, weight.(ps.particles), ps.N, replace=true)
    end
    ps
end

