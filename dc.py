import numpy as np
import time


def rank2_mod(points, mask):
    npoints = points.shape[0]
    if npoints == 1:
        return 0
    else:
        med = np.median(points[:,0])
        idxA = points[:,0] <= med
        A = points[idxA]
        B = points[~idxA]
        rank_A = rank2_mod(A, mask[idxA])
        rank_B = rank2_mod(B, mask[~idxA])
        idxY = np.argsort(points[:,1])
        idxYr = np.zeros_like(idxY)
        idxYr[idxY] = np.arange(idxY.shape[0]) # inverse of idxY
        rank = np.zeros(npoints, dtype=int)
        numA = np.cumsum((idxA*mask)[idxY])[idxYr]
        rank[idxA] = rank_A
        rank[~idxA] = rank_B + numA[~idxA]
        return rank

def rank2(points):
    npoints = points.shape[0]
    if npoints == 1:
        return 0
    else:
        med = np.median(points[:,0])
        idxA = points[:,0] <= med
        A = points[idxA]
        B = points[~idxA]
        rank_A = rank2(A)
        rank_B = rank2(B)
        idxY = np.argsort(points[:,1])
        idxYr = np.zeros_like(idxY)
        idxYr[idxY] = np.arange(idxY.shape[0]) # inverse of idxY
        rank = np.zeros(npoints, dtype=int)
        numA = np.cumsum(idxA[idxY])[idxYr]
        rank[idxA] = rank_A
        rank[~idxA] = rank_B + numA[~idxA]
        return rank

def rankn_mod(points, mask):
    npoints = points.shape[0]
    if npoints == 1:
        return 0
    else:
        if points.shape[1] == 2:
            return rank2_mod(points, mask)
    med = np.median(points[:,0])
    idxA = points[:,0] <= med
    A = points[idxA]
    B = points[~idxA]
    rank_A = rankn_mod(A, mask[idxA])
    rank_B = rankn_mod(B, mask[~idxA])
    rank = np.zeros(npoints, dtype=int)
    rank[idxA] = rank_A
    rank[~idxA] = rank_B + rankn_mod(points[:,1:], idxA*mask)[~idxA] # this needs to be modified.  It computes the total rank, not just the dominated As.
    return rank

def rankn(points):
    npoints = points.shape[0]
    if npoints == 1:
        return 0
    else:
        if points.shape[1] == 2:
            return rank2(points)
    med = np.median(points[:,0])
    idxA = points[:,0] <= med
    A = points[idxA]
    B = points[~idxA]
    rank_A = rankn(A)
    rank_B = rankn(B)
    rank = np.zeros(npoints, dtype=int)
    rank[idxA] = rank_A
    rank[~idxA] = rank_B + rankn_mod(points[:,1:], idxA)[~idxA] # this needs to be modified.  It computes the total rank, not just the dominated As.
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

    npoints = 100
    ndim = 5
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
