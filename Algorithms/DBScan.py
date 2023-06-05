
# import packages
import numpy as np
from Algorithms.BaseAlgo import BaseAlgo

class DBScan(BaseAlgo):
    def __init__(self,
                 data_points: np.array,
                 radios: float,
                 close_bound: int,
                 figures_dir=None):
        super(DBScan, self).__init__(data_points=data_points, figures_dir=figures_dir)

        assert radios > 0 and close_bound > 0, "invalid parameters to DBScan"

        self.data_points = data_points
        self.radios = radios
        self.close_bound = close_bound

    def run(self):

        # distances saved in a N_points * N_points matrix (symmetrical)
        N_points, _ = self.data_points.shape
        distances = self.getDistanceMatrix()

        # determine if each data point is a core point or not and save close points
        core_points = {}
        for i in range(N_points):
            close_points = [j for j, d in enumerate(distances[i, :]) if d and float(d) <= self.radios]
            if len(close_points) >= self.close_bound:
                core_points[i] = close_points

        # recursive assignment of close points to cluster
        def assign(close_points: list, seen: list) -> list:

            if not close_points:
                return []

            close_point = close_points[0]
            close_points = close_points[1:]

            if close_point in seen:
                return []

            seen += [close_point]

            if close_point in core_points:
                new_close_points = core_points[close_point]
                new_close_points = [p for p in new_close_points if p not in seen]
                return [close_point] + assign(close_points, seen) + assign(new_close_points, seen)
            else:
                return [close_point] + assign(close_points, seen)

        # assignment to clusters:
        # for a random core point and assign close points gradually to that cluster
        labels = -np.ones(N_points)
        self.paint(labels, iter=0)

        while core_points:

            # take random core point and assign to new cluster
            core_point, close_points = core_points.popitem()
            new_cluster = int(np.max(np.unique(labels))) + 1
            labels[core_point] = new_cluster

            # skip close points that are already assigned
            close_points = [p for p in close_points if labels[p] < 0]

            # for every close point to the core point, add to the cluster and remove from cores
            # recursive over close points that are also core points
            cluster_points = assign(close_points, seen=[])
            for point in cluster_points:
                labels[point] = new_cluster
                if point in core_points:
                    core_points.pop(point)

        # untouched points are outliers
        new_cluster = int(np.max(np.unique(labels))) + 1
        labels[labels < 0] = new_cluster
        self.labels = labels

        # paint final clustering
        self.paint(labels, iter=1)
        return self










