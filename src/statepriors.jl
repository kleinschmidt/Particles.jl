# structs to represent state transition/occupation priors.

abstract type StatePrior end

"""
    add(::StatePrior, state, n)

Update the sufficient statistics of the state prior, returning a new version of
the state prior and the index corresponding to the state (possibly changed in
the update).
"""
function add(::StatePrior, state, n) end

function log_prior(p::StatePrior)
    cands = candidates(p)
    log_weights = log_prior.(p, cands)
    log_weights .-= StatsFuns.logsumexp(log_weights)
    return log_weights
end

function Base.rand(p::StatePrior)
    weights = Weights(exp.(log_prior(p)))
    sample(candidates(p), weights)
end

function simulate(p::StatePrior, n::Int)
    states = Int[]
    for _ in 1:n
        p, x = add(p, rand(p))
        push!(states, x)
    end
    p, states
end

struct ChineseRestaurantProcess <: StatePrior
    α::Float64
    N::Vector{Float64}
end

ChineseRestaurantProcess(α::Float64) = ChineseRestaurantProcess(α, Vector{Float64}())

candidates(crp::ChineseRestaurantProcess) = 1:length(crp.N)+1
function add(crp::ChineseRestaurantProcess, x::Int, n::Float64=1.)
    N = copy(crp.N)
    if x == length(N)+1
        push!(N, n)
    else
        N[x] += n
    end
    return ChineseRestaurantProcess(crp.α, N), x
end

log_prior(crp::ChineseRestaurantProcess, x::Int) =
    x == length(crp.N)+1 ? log(crp.α) : log(crp.N[x])

marginal_log_prior(crp::ChineseRestaurantProcess) =
    log_prior = sum(lgamma(n) for n in crp.N) + length(crp.N) * log(crp.α)


"""
    StickyCRP <: StatePrior

Represents a "sticky" Chinese Restaurant Process, where there's a constant,
additional probability that the state at time t is the same as time t-1.  That
is, with probability ρ, ``x_t = x_{t-1}``, and with probability ``(1-\rho)``, a
new ``x`` is sampled according to a CRP.

There's some additional book-keeping that's necessary, as "sticking" transitions
are not informative for the CRP prior, but the proportion of transitions that
are same-state are informative when re-sampling the stickiness parameter.  So we
need to track 
1. The number of times a state ``j`` was visited when ``s=0`` (non-sticky).
2. The number of times a state ``j`` was visited when the previous state was 
   also ``j``, _regardless_ of whether sticky or not.

The first is stored on `N`, and the second on `Nsame` (and the value of the last
`x` on `last`).

When generating candidates the value `x=0` is used as a sentinal for sticking.
"""
struct StickyCRP <: StatePrior
    α::Float64
    ρ::Float64                  # probability of "sticking"
    last::Int64
    N::Vector{Float64}          # number of non-sticky occuptations of state
    Nsame::Vector{Float64}      # number of transitions to same state (sticky or not)
end    

StickyCRP(α::Float64, κ::Float64) = StickyCRP(α, κ, 1, Vector{Float64}(), Vector{Float64}())

candidates(crp::StickyCRP) = 0:length(crp.N)+1
add(crp::StickyCRP, x::Int, n::Float64=1.0) = add(crp::StickyCRP, x==0 ? crp.last : x, x==0, n)
function add(crp::StickyCRP, x::Int, sticky::Bool, n::Float64)
    N = copy(crp.N)
    Nsame = copy(crp.Nsame)

    # new component
    if x == length(N)+1
        N = push!(N, 0.)
        Nsame = push!(Nsame, 0.)
    end

    # same transition -> add to Nsame
    if x == crp.last
        Nsame[x] += n
    end

    # sticky -> don't update N
    if !sticky
        N[x] += n
    end

    return StickyCRP(crp.α, crp.ρ, x, N, Nsame), x
end
        

function log_prior(crp::StickyCRP, x::Int)
    if x == 0
        # sticky: ∝ ρ / (1-ρ) * (∑N+α)
        return logit(crp.ρ) + log(sum(crp.N)+crp.α)
    elseif 0 < x ≤ length(crp.N)
        return log(crp.N[x])
    elseif x == length(crp.N)+1
        return log(crp.α)
    else
        throw(ArgumentError("state $x is invalid for sticky CRP with " *
                            "$(length(crp.N)) components (valid values are " *
                            "0..$(length(crp.N)+1)"))
    end
end


################################################################################
# a changepoint state prior

"""
    struct ChangePoint <: StatePrior

A changepoint prior.

# Fields

* logp::Float64 - log-probability of a change
* k::Int - number of states.
* n::Int - number of data points seen so far.
"""
struct ChangePoint <: Particles.StatePrior
    logp::Float64
    k::Int
    n::Int
end

function ChangePoint(p::Float64)
    0 ≤ p ≤ 1 || throw(ArgumentError("ChangePoint probability p must be between 0 and 1 (got p=$p)"))
    ChangePoint(log(p), 0, 0)
end

function add(cp::ChangePoint, state, n=1.)
    state == cp.k || state == cp.k+1 ||
        throw(ArgumentError("ChangePoint state must be same as last ($(cp.k)) or last+1, got $state"))
    ChangePoint(cp.logp, cp.k + (state>cp.k), cp.n+1), state
end

candidates(cp::ChangePoint) =
    max(cp.k,1):cp.k+1

# prior is 1 for starting transition (k=0).  then p for a change, 1-p for stay
log_prior(cp::ChangePoint, state::Int) =
    cp.k == 0 ? 0. :
    state == cp.k ? StatsFuns.log1mexp(cp.logp) : cp.logp

# marginal prior is 1 if there's just one data point.  if > 1, there are k-1
# changepoints, and n-1 opportunities for a changepoint.
marginal_log_prior(cp::ChangePoint) =
    cp.n == 1 ? 0. : cp.logp*(cp.k-1) + StatsFuns.log1mexp(cp.logp)*(cp.n-cp.k)
