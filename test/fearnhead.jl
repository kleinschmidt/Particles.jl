@testset "Fearnhead style mixture model particles" begin

    p = Particle((0.0, 1.0, 1.0, 1.0), (3.0, 1.0, 1.0, 1.0))
    fit!(p, 10., 1)
    putatives(p, 10.)

    @testset "Cutoff for automatic propogation" begin

        using Particles: cutoff
        function cutoff_normalized(ws::Vector{<:Real}, N::Int)
            ws ./= sum(ws)
            total = sum(ws)
            for i in eachindex(ws)
                if total / ws[i] + (i-1) >= N
                    return i, total/(N-i+1)
                end
                total -= ws[i]
            end
        end

        function cutoff_while(ws::Vector{<:Real}, N::Int)
            ws ./= sum(ws)
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

        @test isapprox(cutoff_normalized(ws, 10)[2], cut/tot, atol=1e-10)
        @test isapprox(cutoff_while(ws, 10)[2], cut/tot, atol=1e-10)

        ws = sort(exp.(randn(20)), rev=true)
        keep, cut, tot = cutoff(ws, 10)
        @test isapprox(cutoff_normalized(ws, 10)[2], cut/tot, atol=1e-10)
        @test isapprox(cutoff_while(ws, 10)[2], cut/tot, atol=1e-10)

        ws = [1. for _ in 1:20]
        keep, cut, tot = cutoff(ws, 10)
        @test isapprox(cutoff_normalized(ws, 10)[2], cut/tot, atol=1e-10)
        @test isapprox(cutoff_while(ws, 10)[2], cut/tot, atol=1e-10)

    end

end
