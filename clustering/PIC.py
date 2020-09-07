import numpy as np
import time
from clustering.utils import make_adjacencyW, find_maxima_cluster, preprocess_features, make_graph


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


class PIC(object):
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        xb = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0
