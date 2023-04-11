import numpy as np



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
        idxYr = np.argsort(idxY)
        rank = np.zeros(npoints, dtype=int)
        numA = np.cumsum(idxA[idxY])[idxYr]
        rank[idxA] = rank_A
        rank[~idxA] = rank_B + numA[~idxA]
        return rank

if __name__ == "__main__":

    from matplotlib.pyplot import *

    npoints = 1000
    ndim = 2
    points = (npoints*np.random.random((npoints, ndim)))
    rank = rank2(points)
    scatter(points[:,0], points[:,1], c=rank)
