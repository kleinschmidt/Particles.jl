@testset "Updating particles" begin
    
    @testset "Weight update" begin
        # weights should remain proportional to the conditional posterior p(z|y,x)
        p = InfiniteParticle((0., 1., 0.1, 0.1), 2.0)
        ps = [p]
        for x in 1.:5.
            ps = vcat(putatives.(ps, x)...)
            @test weight.(ps) ≈ marginal_posterior.(ps)
        end
    end
    
end
