using Particles: candidates, instantiate

# original, one-step fit method
function old_fit(p::InfiniteParticle, y, x::Int)
    # first calculate log-prior
    Δlogweight = log_prior(p.stateprior, x)
    # then update sufficient stats and convert x to an index
    stateprior, x = add(p.stateprior, x)
    0 < x ≤ length(p.components)+1 ||
        throw(ArgumentError("can't fit component $x: must be between 0 and " *
                            "$(length(p.components)+1)"))

    components = copy(p.components)
    if x ≤ length(components)
        # likelihood adjustment for old observations
        Δlogweight -= marginal_log_lhood(components[x])
    else
        push!(components, p.prior)
    end
    components[x] = add(components[x], y)
    # likelihood of new observation
    Δlogweight += marginal_log_lhood(components[x])
    weight = exp(log(p.weight) + Δlogweight)

    return InfiniteParticle(components, p, x, weight, p.prior, stateprior)
end


@testset "Updating particles" begin
    
    @testset "putatives vs. direct fit" begin
        p = InfiniteParticle(NormalInverseChisq(0., 1., 0.1, 0.1), 2.0)
        ys = rand(10)

        global ps_put = [p]
        global ps_oldfit = [p]
        global ps_fit = [p]
        for y in ys
            ps_put = reduce(vcat, Particles.instantiate.(putatives(p, y)) for p in ps_put)
            ps_oldfit = reduce(vcat, old_fit.(Ref(p), y, candidates(p.stateprior)) for p in ps_fit)
            ps_fit = reduce(vcat, fit.(Ref(p), y, candidates(p.stateprior)) for p in ps_oldfit)
            @test components.(ps_put) == components.(ps_fit) == components.(ps_oldfit)
        end
    end
        
    @testset "Weight update" begin
        # weights should remain proportional to the conditional posterior p(z|y,x)
        p = InfiniteParticle(NormalInverseChisq(0., 1., 0.1, 0.1), 2.0)
        ps = [p]
        for x in 1.:5.
            ps = Particles.instantiate.(vcat(collect.(putatives.(ps, x))...))
            @test weight.(ps) ≈ marginal_posterior.(ps)
        end
    end

    
end
