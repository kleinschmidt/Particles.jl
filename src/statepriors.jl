# structs to represent state transition/occupation priors.

abstract type StatePrior end

struct ChineseRestaurantProcess
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
    return ChineseRestaurantProcess(crp.α, N)
end

log_prior(crp::ChineseRestaurantProcess, x::Int) =
    x == length(crp.N)+1 ? log(crp.α) : log(crp.N[x])

marginal_log_prior(crp::ChineseRestaurantProcess) =
    log_prior = sum(lgamma(n) for n in crp.N) + length(crp.N) * log(crp.α)



struct StickyCRP
    α::Float64
    ρ::Float64                  # probability of "sticking"
    last::Int64
    N::Vector{Float64}          # number of transitions to each from different
    Nsame::Vector{Float64}      # number of transitions to same state
end    

StickyCRP(α::Float64, κ::Float64) = StickyCRP(α, κ, 0, Vector{Float64}(), Vector{Float64}())

candidates(crp::StickyCRP) = 1:length(crp.N)+1
function add(crp::StickyCRP, x::Int, n::Float64=1.0)
    Nsame = crp.Nsame
    if x == crp.last
        Nsame = copy(crp.Nsame)
        Nsame[x] += n
        return StickyCRP(crp.α, crp.ρ, x, crp.N, Nsame)
    elseif x == length(crp.N) + 1
        return StickyCRP(crp.α, crp.ρ, x, push!(copy(crp.N), n), push!(copy(crp.Nsame), 0.))
    else
        N = copy(crp.N)
        N[x] += n
        return StickyCRP(crp.α, crp.ρ, x, N, crp.Nsame)
    end
end
        
        
function log_prior(crp::StickyCRP, x::Int)
    # prior is (1-ρ) * CRP prior, plus ρ * δ(x, crp.last)
    #
    # CRP prior is ∝ N_j if j∈1..K or α if j=K+1.
    # the constant of proportionality is ∑N_j + α.
    #
    # total is (1-ρ)*N_j + ρ*δ(x, x_{n-1}) ∝ N_j + ρ/(1-ρ) δ(x, x_{n-1})
    #
    # p(s=1 | x_n = x_n-1) = p(s, x_n=x_n-1) / p(x_n=x_n-1)
    #                      = p(x_n=x_n-1 | s) p(s) / p(x_n=x_n-1)
    #
    # p(s=1) = ρ
    # p(x_n=x_n-1 | s=1) = 1
    # p(x_n=x_n-1 | s=0) = ...?
    #
    # p(x_n = k | ρ, x_n-1) = ∑_s p(x_n=k | x_n-1, s) p(s | ρ)
    #                       = δ(x_n, x_n-1) ρ + N_k/(N+α) (1-ρ)
    #
    # twist is that the N for CRP prior needs to adjust for the possibility that
    # some self-transitions were due to not sticking.  given that there's a
    # self-transition, the expected proportion of non-sticks is (1-ρ).  so the
    # effective N is N + (1-ρ)Nsame.
    #
    # likewise the total count (for non-sticking) is N' = ∑N + (1-ρ)∑Nsame + α.
    # the contribution if you don't stick is N'*(1-ρ)...
    ρ = crp.ρ
    if x == length(crp.N)+1
        logp = log(crp.α) + log(1-ρ)
    else
    logp = log(crp.N[x] + crp.Nsame[x]*(1-ρ)) + log(1-ρ)
    if x == crp.last
        logp = logsumexp(logp, log(ρ))
    end
    return logp
end
