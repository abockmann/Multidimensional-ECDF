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
        rank[~idxA] = rank_B + rankn(points[:,1:])[~idxA]
        return rank

if __name__ == "__main__":

    # type "%matplotlib qt" in jupyter input for interactive plots

    from matplotlib.pyplot import *

    npoints = 1000
    ndim = 3
    points = (npoints*np.random.random((npoints, ndim)))
    rank = rankn(points)
    fig = figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=rank)
