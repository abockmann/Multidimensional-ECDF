import numpy as np

def _rank2(points, mask=None):
    N = points.shape[0]
    N2 = N//2
    if N == 1:
        return 0
    else:
        idx = np.argpartition(points[:,0], N2)
        idxA_ = idx[:N2]
        idxA = np.zeros(N, dtype=bool)
        idxA[idxA_] = True
        if mask is not None:
            NAm = np.sum(idxA & mask)
            points_reduced = np.vstack((points[idxA & mask], points[~idxA & ~mask]))
        else:
            NAm = np.sum(idxA)
            points_reduced = np.vstack((points[idxA], points[~idxA]))
        count_points = np.zeros(points_reduced.shape[0], dtype=bool)
        count_points[:NAm] = True
        idxY = np.argsort(points_reduced[:,1])
        idxYr = np.zeros_like(idxY)
        idxYr[idxY] = np.arange(idxY.shape[0]) # inverse of idxY
        count_points = count_points[idxY]
        numA = np.cumsum(count_points)[idxYr]
        rank = np.zeros(N, dtype=int)
        if mask is not None:
            rank[idxA] = _rank2(points[idxA], mask[idxA])
            rank[~idxA] = _rank2(points[~idxA], mask[~idxA])
            rank[~idxA & ~mask] += numA[NAm:]
        else:
            rank[idxA] = _rank2(points[idxA])
            rank[~idxA] = _rank2(points[~idxA])
            rank[~idxA] += numA[NAm:]
        return rank

def rankn(points, mask=None):
    N = points.shape[0]
    N2 = N//2
    if mask is None:
        mask = np.ones(N, dtype=bool)
        first_call = True
    else:
        first_call = False
    if N == 1:
        return 0
    if points.shape[1] == 2:
        if first_call:
            return _rank2(points)
        else:
            return _rank2(points, mask)
    idx = np.argpartition(points[:,0], N2)
    idxA_ = idx[:N2]
    idxA = np.zeros(N, dtype=bool)
    idxA[idxA_] = True
    rank = np.zeros(N, dtype=int)
    rank[idxA] = rankn(points[idxA], mask[idxA])
    rank[~idxA] = rankn(points[~idxA], mask[~idxA]) + rankn(points[:,1:], idxA*mask)[~idxA]
    return rank


if __name__ == "__main__":

    import time

    def rank_naive(points):
        # a naive O(N^2) implementation for validation
        N = points.shape[0]
        rank = np.zeros(N, dtype=int)
        for i in range(N):
            rank[i] = np.sum(np.all(points[i] > points, axis=1))
        return rank

    N = 10000
    ndim = 2
    points = np.random.random((N, ndim))
    s = time.time()
    rank = rankn(points)
    print(f'Multidimensional divide & conquer: {time.time() - s} s')

    s = time.time()
    rank_n = rank_naive(points)
    print(f'Naive implementation: {time.time() - s} s')

    assert all((rank - rank_n) == 0)
