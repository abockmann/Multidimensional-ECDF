import numpy as np
import time


def rank2(points, mask):
    npoints = points.shape[0]
    if npoints == 1:
        return 0
    else:
        med = np.median(points[:,0])
        idxA = points[:,0] <= med
        A = points[idxA]
        B = points[~idxA]
        rank_A = rank2(A, mask[idxA])
        rank_B = rank2(B, mask[~idxA])
        idxY = np.argsort(points[:,1])
        idxYr = np.zeros_like(idxY)
        idxYr[idxY] = np.arange(idxY.shape[0]) # inverse of idxY
        rank = np.zeros(npoints, dtype=int)
        numA = np.cumsum((idxA*mask)[idxY])[idxYr]
        rank[idxA] = rank_A
        rank[~idxA] = rank_B + numA[~idxA]
        return rank

def rankn(points, mask=None):
    npoints = points.shape[0]
    if mask is None:
        mask = np.ones(npoints, dtype=bool)
    if npoints == 1:
        return 0
    else:
        if points.shape[1] == 2:
            return rank2(points, mask)
    med = np.median(points[:,0])
    idxA = points[:,0] <= med
    A = points[idxA]
    B = points[~idxA]
    rank_A = rankn(A, mask[idxA])
    rank_B = rankn(B, mask[~idxA])
    rank = np.zeros(npoints, dtype=int)
    rank[idxA] = rank_A
    rank[~idxA] = rank_B + rankn(points[:,1:], idxA*mask)[~idxA] # this needs to be modified.  It computes the total rank, not just the dominated As.
    return rank
    
def naive(points):
    # a naive O(n^2) Implementation for validation
    npoints = points.shape[0]
    rank = np.zeros(npoints, dtype=int)
    for i in range(npoints):
        rank[i] = np.sum(np.all(points[i] > points, axis=1))
    return rank


if __name__ == "__main__":

    # type "%matplotlib qt" in jupyter input for interactive plots

    from matplotlib.pyplot import *

    npoints = 1000
    ndim = 4
    points = (npoints*np.random.random((npoints, ndim)))
    s = time.time()
    rank = rankn(points)
    print(time.time() - s)
    fig = figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(points[:,0], points[:,1], points[:,2], c=rank)

    s = time.time()
    rank_naive = naive(points)
    print(time.time() - s)
    #ax.scatter(points[:,0], points[:,1], points[:,2], c=rank_naive)

    # 2d algorithm works perfectly, but not 3d
    assert all((rank - rank_naive) == 0)
