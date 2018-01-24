# a gibbs sampler to compare.

# general algorithm:
# * pull out data point x_i
# * compute p(z_i = k | x_i) \propto p(x_i | z_i=k) p(z_i=k)
# * sample z_i from this.


"""
    GibbsCRP

Non-parametric clustering via Gibbs sampling with a Chinese Restaurant Process.

# Constructors

    GibbsCRP(prior, α, x)

Initialize all data assigned to one component.

# Fields

* `prior::NormalInverseChisq`: Prior for each cluster
* `logα::Float64`: (log) pseudo observations reserved for a new cluster
* `components::Vector{Component}`: Clusters
* `empties::Vector{Int}`: Keeps track of which indices are empty and can be
   recycled
* `assignments::Vector{Int}`: Current cluster assignments for each data point
* `data::Vector{Float64}`: Data points to cluster

"""
mutable struct GibbsCRP
    prior::NormalInverseChisq
    logα::Float64
    components::Vector{Component}
    empties::Vector{Int}
    assignments::Vector{Int}
    data::Vector{Float64}
end


GibbsCRP(prior::NormalInverseChisq, α::Float64, x::Vector{Float64}) =
    GibbsCRP(prior,
             log(α),
             [Component(prior, suffstats(Normal, x)), Component(prior)],
             [2],
             ones(Int, length(x)),
             x)

function StatsBase.sample!(gc::GibbsCRP, i::Int)
    x = gc.data[i]
    old_comp = sub(gc.components[gc.assignments[i]], x)
    gc.components[gc.assignments[i]] = old_comp

    # recycle if empty
    isempty(old_comp) && push!(gc.empties, gc.assignments[i])

    log_probs = 
        [isempty(comp) ?
           -Inf :
           logpdf(comp, x) + log(nobs(comp))
         for comp
         in gc.components]

    log_probs[gc.empties[1]] = logpdf(Component(gc.prior), x) + gc.logα

    new_k = sample(Weights(exp.(log_probs)))
    gc.components[new_k] = add(gc.components[new_k], x)
    gc.assignments[i] = new_k

    # clean up empties
    if new_k == gc.empties[1]
        shift!(gc.empties)
        gc.empties
        if isempty(gc.empties)
            push!(gc.components, Component(gc.prior))
            push!(gc.empties, length(gc.components))
        end
    end

    return gc
end

function StatsBase.sample!(gc::GibbsCRP)
    for i in 1:length(gc.data)
        sample!(gc, i)
    end
    return gc
end
