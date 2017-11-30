@testset "Fearnhead style mixture model particles" begin

    p = Particle((0.0, 1.0, 1.0, 1.0), (3.0, 1.0, 1.0, 1.0))
    fit!(p, 10., 1)
    putatives(p, 10.)

end
