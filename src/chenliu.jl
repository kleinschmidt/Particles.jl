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
        println(io, "Particle filter with $n (up to $(ps.N)) particles:")
    else
        println(io, "Particle filter with $n particles:")
    end
    show(IOContext(io, :compact=>true), ps.particles)
end

# initialize with full population of empty particles, because this method
# doens't benefit from redundancy in the same way as the Fearnhead method does
ChenLiuParticles(n::Int, priors::Union{Tuple,<:Distribution}...; rejuv::Float64=50.) =
    ChenLiuParticles([Particle(priors...) for _ in 1:n], n, rejuv)
ChenLiuParticles(n::Int, prior::Union{Tuple,<:Distribution}, stateprior::T; rejuv::Float64=50.) where T<:StatePrior =
    ChenLiuParticles([InfiniteParticle(prior, stateprior) for _ in 1:n], n, rejuv)

function propogate_chenliu(p::P, y) where P<:AbstractParticle
    ps = collect(putatives(p, y))
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

function fit!(ps::ChenLiuParticles, y)
    map!(p->propogate_chenliu(p, y), ps.particles, ps.particles)
    # rejuvinate if necessary
    ws = weight.(ps.particles)
    cv = coefvar(ws)
    if cv > ps.rejuvination_threshold
        @debug "  Rejuvinating ($cv > $(ps.rejuvination_threshold))"
        ps.particles = weight.(wsample(ps.particles, weight.(ps.particles), ps.N, replace=true), 1/ps.N)
    else
        # normalize weights to prevent underflow
        # TODO: use log-weight instead to avoid this altogether
        total_w = sum(ws)
        ps.particles = weight.(ps.particles, ws ./ total_w)
    end
    ps
end

