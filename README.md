# Particles

[![Build Status](https://travis-ci.org/kleinschmidt/Particles.jl.svg?branch=master)](https://travis-ci.org/kleinschmidt/Particles.jl)

[![codecov.io](http://codecov.io/github/kleinschmidt/Particles.jl/coverage.svg?branch=master)](http://codecov.io/github/kleinschmidt/Particles.jl?branch=master)

This package implements three different Bayesian non-parametric methods for
inferring the number of clusters in a mixture model, and cluster assignemnts.

1. A Gibbs sampler which makes multiple passes over the data (batch).
2. A particle filter using the resampling scheme described by [Fearnhead
   (2004)](https://doi.org/10.1023/B:STCO.0000009418.04621.cd) (online).
3. A particle filter using the resampling scheme of [Chen and Liu
   (2000)](https://doi.org/10.1111/1467-9868.00246) (online).

All three use a Dirichlet-process mixture-of-gaussians model, with a conjugate
Normal-Inverse Chi-squred prior for the parameters (mean and variance).  In
principle the code could be easily generalize to use any of the distributions in
[ConjugatePriors.jl](https://github.com/JuliaStats/ConjugatePriors.jl), but for
now the Guassian/NIX2 is hard-coded.

