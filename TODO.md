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

## Optimization

### Putatives

There's no need to keep anything other than the putatives for the old
particles.  And in fact you might be able to represent the whole _population_
more efficiently if you keep a vector of all the components and then just index
into that...that might be too fiddly and not buy you too much though.

I guess the question always is, what's the problem this optimization is trying
to solve?  The run-away copying of vectors that causes some out of control
memory use.  But that's not the major bottleneck except for the really easy 1-D
cases.

### Refactoring filters

The filters should be re-factored into the filter itself and the
propogation/resampling method.
