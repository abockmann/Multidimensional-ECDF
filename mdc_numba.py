import numpy as np
from numba import njit, prange

@njit
def _rank2(points, mask=None):
    N = points.shape[0]
    N2 = N//2
    if N == 1:
        return np.array([0])
    else:
        idx = np.argsort(points[:,0])
        idxA_ = idx[:N2]
        idxA = np.full(N, False)
        idxA[idxA_] = True
        if mask is not None:
            NAm = np.sum(idxA & mask)
            points_reduced = np.vstack((points[idxA & mask], points[~idxA & ~mask]))
        else:
            NAm = np.sum(idxA)
            points_reduced = np.vstack((points[idxA], points[~idxA]))
        count_points = np.full(points_reduced.shape[0], False)
        count_points[:NAm] = True
        idxY = np.argsort(points_reduced[:,1])
        idxYr = np.zeros_like(idxY)
        idxYr[idxY] = np.arange(idxY.shape[0]) # inverse of idxY
        count_points = count_points[idxY]
        numA = np.cumsum(count_points)[idxYr]
        rank = np.full(N, 0)
        if mask is not None:
            for i in prange(3):
                if i == 0:
                    rank[idxA] = _rank2(points[idxA], mask[idxA])
                elif i == 1:
                    rank[~idxA] = _rank2(points[~idxA], mask[~idxA])
                elif i == 2:
                    rank[~idxA & ~mask] += numA[NAm:]
        else:
            for i in prange(3):
                if i == 0:
                    rank[idxA] = _rank2(points[idxA])
                elif i == 1:
                    rank[~idxA] = _rank2(points[~idxA])
                elif i == 2:
                    rank[~idxA] += numA[NAm:]
        return rank

@njit
def rankn(points, mask=None):
    N = points.shape[0]
    N2 = N//2
    if mask is None:
        mask = np.full(N, True)
        first_call = True
    else:
        first_call = False
    if N == 1:
        return np.array([0])
    if points.shape[1] == 2:
        if first_call:
            return _rank2(points)
        else:
            return _rank2(points, mask)
    idx = np.argsort(points[:,0])
    idxA_ = idx[:N2]
    idxA = np.full(N, False)
    idxA[idxA_] = True
    rank = np.full(N, 0)
    for i in prange(3):
        if i == 0:
            rank[idxA] = rankn(points[idxA], mask[idxA])
        elif i == 1:
            rank[~idxA] = rankn(points[~idxA], mask[~idxA])
        elif i == 2:
            rank[~idxA] += rankn(points[:,1:], idxA*mask)[~idxA]
    return rank


if __name__ == "__main__":

    import time

    @njit
    def rank_naive(points):
        # a naive O(N^2) implementation for validation
        ndim = points.shape[1]
        N = points.shape[0]
        rank = np.full(N, 0)
        for i in prange(N):
            cond = np.sum(points[i] > points, axis=1)
            rank[i] = np.sum(cond == ndim)
        return rank

    N = 10000
    ndim = 2
    points = np.random.random((N, ndim))
    i = np.argsort(points[:,-1])
    points = points[i]
    s = time.time()
    rank = rankn(points)
    print(f'Multidimensional divide & conquer: {time.time() - s} s')

    s = time.time()
    rank_n = rank_naive(points)
    print(f'Naive implementation: {time.time() - s} s')

    assert all((rank - rank_n) == 0)
