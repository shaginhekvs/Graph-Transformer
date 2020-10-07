from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ri
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, SpectralClustering
import itertools
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io 


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def accuracy_clustering(y_true, y_pred):
    
    # Ordering labels
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    
    scores = []
    
    # Try all the possible permutations
    permutations = list(itertools.permutations(labels))
    for perm in permutations:
        y_permuted = np.zeros_like(y_true)
        for i,k in enumerate(perm):
            y_permuted[y_pred==k] = labels[i]
        score = accuracy_score(y_true, y_permuted)
        scores.append(score)
    
    return max(scores)



def print_evaluation_from_embeddings(y_true, embeddings, K=5):

    y = KMeans(K, random_state=42).fit_predict(embeddings)

    acc_spec = accuracy_clustering(y_true, y)
    pu_spec        = purity_score(y_true, y)
    nmi_score_spec = nmi(y_true.ravel(), y.ravel())#, average_method='geometric')
    ri_score_spec  = ri(y_true.ravel(), y.ravel())

    print('Accuracy', acc_spec, 'Purity', pu_spec, 'NMI', nmi_score_spec, 'RI', ri_score_spec)
    return acc_spec