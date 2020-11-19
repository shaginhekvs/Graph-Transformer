
from sklearn.neighbors import kneighbors_graph


def build_kneighbors(X, knn=True, n_neighbors=20):
    if knn:
        A = kneighbors_graph(X, n_neighbors, include_self=False)
        A = np.array(A.todense())
        A = (A + A.T)/2
        A = (A >0).astype(int)
    else:
        A = pairwise_kernels(X, metric='rbf', gamma=1)
    return A


