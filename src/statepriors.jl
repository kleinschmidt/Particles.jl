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
