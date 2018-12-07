using Base.Iterators
using Particles: instantiate
using Distributions, ConjugatePriors

@testset "Labeled observations" begin
    ys = rand(MixtureModel([Normal(-2.), Normal(2.)]), 20)

    labels = ifelse.(ys .> 0, 1, 2)

    ylab = Labeled.(ys, labels)

    prior = NormalInverseChisq(0., 1., 2., 3.)

    p = InfiniteParticle(Component.([prior, prior]),
                         nothing,
                         0,
                         1.,
                         Component(prior),
                         NStatePrior([1., 1.]))

    @testset "putatives with labeled" begin
        lab1, lab2 = instantiate.(putatives(p, ys[1]))
        @test lab1.components == putatives(p, Labeled(ys[1], 1))[1].components
        @test lab2.components == putatives(p, Labeled(ys[1], 2))[1].components

        lab1m, lab2m = instantiate.(putatives(p, Labeled(ys[1])))
        @test lab1m.components == lab1.components
        @test lab2m.components == lab2.components
    end

    @testset "filter equivalence with missing" begin
        Random.seed!(1)
        fil1 = reduce(fit!, ys, init=FearnheadParticles([p], 10))

        Random.seed!(1)
        fil2 = reduce(fit!, Labeled.(ys, missing), init=FearnheadParticles([p], 10))

        @test getfield.(fil1.particles, :components) ==
            getfield.(fil2.particles, :components)
    end

    @testset "filtering with labels" begin
        fil = reduce(fit!, ylab, init=FearnheadParticles([p], 10))
        # only one particle:
        @test length(fil.particles) == 1
        pp = first(fil.particles)
        @test nobs(pp) == length(ylab)
        @test nobs.(pp.components) == [sum(labels.==1), sum(labels.==2)]

        # same results if missing labels are before or after:
        some_unlab = Labeled.([-1., 0, 1], missing)
        fil_before = reduce(fit!, append!(copy(some_unlab), ylab),
                            init=FearnheadParticles([p], 10))
        fil_after = reduce(fit!, prepend!(copy(some_unlab), ylab),
                           init=FearnheadParticles([p], 10))

        # test equality of component parameters
        for (p1,p2) in zip(fil_before.particles, fil_after.particles)
            for (c1,c2) in zip(p1.components, p2.components)
                @test all(params(c1) .≈ params(c2))
            end
        end
        
    end
end
