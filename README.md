# Multidimensional-ECDF
A Python/numpy implementation of the algorithm described in section 2.1.1 of Jon Louis Bentley, "Multidimensional Divide-and-Conquer" (1980)

The rankn function can be used to compute the Empirical CDF (just divide the rank by the number of points) for a collection of multidimensional points.  The algorithm is much faster than brute force for two (maybe three) dimensions and a large number of points, but the CPU time increases very fast with an increasing number of dimensions, unlike the naive approach.
The main obstacle when it comes to efficiency seems to be the recursive nature, which breaks the work up in lots of small pieces, which Python execute sequentially.

Author: Arne Bøckmann
