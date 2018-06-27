using Particles: ChineseRestaurantProcess, marginal_log_prior, candidates

@testset "Priors on states" begin
    @testset "Chinese restaurant process" begin

        crp = ChineseRestaurantProcess(0.5)
        @test crp.N == Float64[]
        @test crp.α == 0.5

        @test candidates(crp) == 1:1

        crp1, _ = add(crp, 1)
        @test crp1.N == [1.]
        @test crp1.α == crp.α
        @test candidates(crp1) == 1:2

        

        crp11, _ = add(crp1, 1)
        @test crp11.N == [2.]

        crp111, _ = add(crp11, 1, 0.5)
        @test crp111.N == [2.5]

        @test_throws BoundsError add(crp, 2)
        @test_throws BoundsError add(crp1, 3)

        crp1112, _ = add(crp111, 2)
        @test crp1112.N == [2.5, 1.]
        @test candidates(crp1112) == 1:3
        
    end
end
