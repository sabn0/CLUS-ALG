
# import packages
import numpy as np
from Algorithms.BaseAlgo import BaseAlgo


class KMeans(BaseAlgo):
    def __init__(self,
                 data_points: np.array,
                 K: int,
                 figures_dir=None,
                 max_iter=10,
                 ):
        super(KMeans, self).__init__(data_points=data_points, figures_dir=figures_dir)

        assert K > 0, "invalid number of centroids provided: {}".format(K)
        self.max_iter = max_iter
        self.K = K

    def run(self):

        # expectation - maximization implementation for n'dim data
        N_points, data_dim = self.data_points.shape

        # create initial centroids
        centroids = np.random.rand(self.K, data_dim)
        labels = np.random.choice(self.K, N_points)

        for iter in range(self.max_iter):

            # paint process
            self.paint(labels, iter, centroids=centroids)

            # expectation step :
            # calculate (euc) distance from each data_point to every centroid
            # and find the closest centroid for the data point
            for i, data_point in enumerate(self.data_points):
                distances = []
                for centroid in centroids:
                    distances += [self.euclidean_distance(data_point, centroid)]

                best = np.argmin(distances)
                labels[i] = best

            # maximization step :
            # find new centroids that "maximize" the found labels
            # (a centroid should minimize the overall distance from its points, be the average of data points)
            old_centroids = centroids.copy()
            for i, centroid in enumerate(centroids):

                centroid_data = np.array([d_p for label, d_p in zip(labels, self.data_points) if label == i])
                new_centroid = np.mean(centroid_data, axis=0)
                centroids[i] = new_centroid

            # break if centroids don't change
            if np.sum(np.abs(old_centroids-centroids)) < float('1e-10'):
                break

        self.labels = labels
        self.makeGIF()
        return self

