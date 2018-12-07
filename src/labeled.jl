struct Labeled{T}
    obs::T
    label::Union{Missing,Int}
end

Base.broadcastable(x::Labeled) = Ref(x)

Labeled(x) = Labeled(x, missing)

putatives(p::InfiniteParticle, x::Labeled) =
    x.label === missing ? putatives(p, x.obs) : (fit(p, x.obs, x.label), )
