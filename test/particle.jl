using Particles: candidates, instantiate

@testset "Updating particles" begin
    
    @testset "putatives vs. direct fit" begin
        p = InfiniteParticle(NormalInverseChisq(0., 1., 0.1, 0.1), 2.0)
        ys = rand(10)

        global ps_put = [p]
        global ps_fit = [p]
        for y in ys
            ps_put = reduce(vcat, Particles.instantiate.(putatives(p, y)) for p in ps_put)
            ps_fit = reduce(vcat, fit.(Ref(p), y, candidates(p.stateprior)) for p in ps_fit)
            @test components.(ps_put) == components.(ps_fit)
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
