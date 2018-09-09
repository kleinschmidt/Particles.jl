using Particles: marginal_log_prior, log_prior, candidates, simulate, add

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

    @testset "Sticky CRP" begin

        @testset "StickyCRP with ρ=0 == CRP" begin
            scrp = StickyCRP(0.5, 0.0)
            crp = ChineseRestaurantProcess(0.5)

            Random.seed!(1)
            scrp_sim, scrp_states = simulate(scrp, 100)
            Random.seed!(1)
            crp_sim, crp_states = simulate(crp, 100)

            @test scrp_states == crp_states
            @test scrp_sim.N == crp_sim.N

            @test log_prior.(Ref(crp_sim), candidates(crp_sim)) ==
                log_prior.(Ref(scrp_sim), candidates(scrp_sim))[2:end]

        end

        crp = StickyCRP(0.5, 0.9)
        crp1, s1 = add(crp, 1, 0.5)
        crp0, s0 = add(crp, 0, 0.5)

        # state is same after `add` with sticky or not
        @test s1 == s0
        @test crp1.N == [0.5]
        # ... but counts are not.
        @test crp0.N == [0.]

    end
end
