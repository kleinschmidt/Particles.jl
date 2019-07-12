@testset "Fearnhead style mixture model particles" begin

    @testset "Cutoff for automatic propogation" begin

        using Particles: cutoff_ascending

        # a stub to replace the original cutoff using cutoff_ascending
        function cutoff(ws, N)
            ws = sort(ws)
            i_keep, w = cutoff_ascending(ws, N)
            return length(ws) - i_keep + 2, w, sum(ws)
        end

        function cutoff_normalized(ws::Vector{<:Real}, N::Int)
            ws = ws ./ sum(ws)
            total = sum(ws)
            for i in eachindex(ws)
                if total / ws[i] + (i-1) >= N
                    return i, total/(N-i+1)
                end
                total -= ws[i]
            end
        end

        function cutoff_while(ws::Vector{<:Real}, N::Int)
            ws = ws ./ sum(ws)
            total = sum(ws)
            i = 1
            while total / ws[i] + (i-1) < N
                total, i = total - ws[i], i+1
                i > length(ws) && error("infinite loop??")
            end
            i, total / (N-i+1)
        end

        ws = sort([1e10, exp.(randn(19))...], rev=true)
        keep, cut, tot = cutoff(ws, 10)
        # @show cutoff_normalized(ws,10)[2], cut, tot,  cut/tot

        # check correctness of the algorithm:  should be the case that
        # \sum_i=1^M min( c*w_i, 1 ) = N
        @test isapprox(sum(min.(ws ./ cut, 1)), 10, atol=1e-5)

        @test isapprox(cutoff_normalized(ws, 10)[2], cut/tot, atol=1e-10)
        @test isapprox(cutoff_while(ws, 10)[2], cut/tot, atol=1e-10)

        ws = sort(exp.(randn(20)), rev=true)
        keep, cut, tot = cutoff(ws, 10)
        @test isapprox(cutoff_normalized(ws, 10)[2], cut/tot, atol=1e-10)
        @test isapprox(cutoff_while(ws, 10)[2], cut/tot, atol=1e-10)

        # when weights all equal, resample all with weight 1/N
        ws = [1. for _ in 1:20]
        keep, cut, tot = cutoff(ws, 10)
        @test isapprox(cutoff_normalized(ws, 10)[2], cut/tot, atol=1e-10)
        @test isapprox(cutoff_while(ws, 10)[2], cut/tot, atol=1e-10)

        @test keep == 1         # first particle NOT kept
        @test cut/tot ≈ 1/10

        # when weights are all equal or zero, also resample all with weight 1/N
        # ...actually, in THIS case you effectively only HAVE N particles withn
        # non-zero weight so you should keep them all, which is what
        # cutoff-ascending gives you (unlike the old solution).
        ws = [i > 10 ? 0. : 1. for i in 1:20]
        keep, cut, tot = cutoff(ws, 10)
        @test keep == 11

        ws = [i > 10 ? 0. : rand() for i in 1:20]
        keep, cut, tot = cutoff(ws, 10)
        @test keep == 11

    end

    @testset "stratified sampling" begin

        Random.seed!(1999)
        w = exp.(2*rand(100))
        w ./= sum(w)
        x = collect(1:100)
        
        for n in [1, 8, 64, 90, 99]
            for _ in 1:100
                x_samp = Particles.sample_stratified(x, n, w)
                @test length(x_samp) ≤ n
            end
        end
                
    end

    

end
