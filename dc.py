import numpy as np
import time



def rank2(points, mask):
    N = points.shape[0]
    N2 = N//2
    if N == 1:
        return 0
    else:
        idx = np.argpartition(points[:,0], N2)
        idxA_ = idx[:N2]
        idxA = np.zeros(N, dtype=bool)
        idxA[idxA_] = True
        rank_A = rank2(points[idxA], mask[idxA])
        rank_B = rank2(points[~idxA], mask[~idxA])
        N_split = np.sum(idxA & mask)
        points_reduced = np.vstack((points[idxA & mask], points[~idxA & ~mask]))
        count_points = np.zeros(points_reduced.shape[0], dtype=bool)
        count_points[:N_split] = True
        idxY = np.argsort(points_reduced[:,1])
        idxYr = np.zeros_like(idxY)
        idxYr[idxY] = np.arange(idxY.shape[0]) # inverse of idxY
        count_points = count_points[idxY]
        numA = np.cumsum(count_points)[idxYr]
        rank = np.zeros(N)
        rank[idxA] = rank_A
        rank[~idxA] = rank_B
        rank[~idxA & ~mask] += numA[N_split:]
        return rank

def rankn(points, mask=None):
    N = points.shape[0]
    N2 = N//2
    if mask is None:
        mask = np.ones(N, dtype=bool)
    if N == 1:
        return 0
    else:
        if points.shape[1] == 2:
            return rank2(points, mask)
    idx = np.argpartition(points[:,0], N2)
    idxA_ = idx[:N2]
    idxA = np.zeros(N, dtype=bool)
    idxA[idxA_] = True
    rank_A = rankn(points[idxA], mask[idxA])
    rank_B = rankn(points[~idxA], mask[~idxA])
    rank = np.zeros(N, dtype=int)
    rank[idxA] = rank_A
    rank[~idxA] = rank_B + rankn(points[:,1:], idxA*mask)[~idxA] # this needs to be modified.  It computes the total rank, not just the dominated As.
    return rank
    
def naive(points):
    # a naive O(n^2) Implementation for validation
    N = points.shape[0]
    rank = np.zeros(N, dtype=int)
    for i in range(N):
        rank[i] = np.sum(np.all(points[i] > points, axis=1))
    return rank



if __name__ == "__main__":



    # type "%matplotlib qt" in jupyter input for interactive plots

    from matplotlib.pyplot import *

    N = 1000
    ndim = 4
    points = (N*np.random.random((N, ndim)))
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
