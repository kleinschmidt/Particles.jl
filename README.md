# Particles

[![Build Status](https://travis-ci.org/kleinschmidt/Particles.jl.svg?branch=master)](https://travis-ci.org/kleinschmidt/Particles.jl)

[![Coverage Status](https://coveralls.io/repos/kleinschmidt/Particles.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/kleinschmidt/Particles.jl?branch=master)

[![codecov.io](http://codecov.io/github/kleinschmidt/Particles.jl/coverage.svg?branch=master)](http://codecov.io/github/kleinschmidt/Particles.jl?branch=master)

## Interface

Given a series of data, we want to make inferences about two kinds of latent
variables: cluster labels and cluster parameters.  The Fearnhead approach only
uses particles to represent the cluster assignments, and relies on conjugate
priors for the cluster parameters to get the posteriors of the cluster
parameters conditional on the assignments.  There are three kinds of operations
you might want to do:

1. **filter** one observation (get the posterior distribution of the cluster
   assignment after absorbing one observation).  $p(z_T | x_{1:T})$.  This uses
   only past and present data to infer the state for present data.
2. **smooth** a series of observations (get the posterior distribution of the
   _sequence_ of cluster assignments after a series of observation).  $p(z_{1:T}
   | x_{1:T})$.  This uses "future" data, in the sense that inferences about the
   state $z_k$ at $k<T$ incorporates information from future data at times
   $k+1:T$.
3. **predict** future data or states: $p(x_{T+1} | x_{1:T})$ or $p(z_{T+1} |
   x_{1:T})$.

## Nonparametrics

The Fearnhead approach can be used to do non-parametric clustering, where the
number of clusters is unknown.  This makes the filtering/smoothing a little
tricky, since the cluster labels might not be consistent across particles.  So
just marginalizing across the particles doesn't make sense.  But that's a bridge
to cross when we come to it.  One possibility is to return a similarity matrix
sort of thing, which is the probability that any pair of observations belongs to
the same cluster.  _That_ can be marginalized over (and compared to the
"correct" clustering solution).
