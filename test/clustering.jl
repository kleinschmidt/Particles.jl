@testset "Clustering metrics" begin
    x = rand(50)

    for ps_type in (FearnheadParticles, ChenLiuParticles)
        @testset "$ps_type" begin
            ps = ps_type(100, NormalInverseChisq(0., 1, 2, 2), ChineseRestaurantProcess(1.))
            filter!(ps, x, false)
            
            asgn = assignments(ps)
            @test size(asgn) == (50, 100)
            @test eltype(asgn) == Int
            @test vec(maximum(asgn, dims=1)) == ncomponents.(ps.particles)
            @test all(minimum(asgn, dims=1) .== 1)

            asgn_sim = assignment_similarity(ps)
            @test size(asgn_sim) == (50, 50)
            @test issymmetric(asgn_sim)
            @test all(0 .≤ asgn_sim .≤ 1)
            @test all(diag(asgn_sim) .== 1)
        end
    end
end
