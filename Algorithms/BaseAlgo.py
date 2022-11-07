
# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod

class BaseAlgo:
    def __init__(self, data_points: np.array, figures_dir=None):

        # data points of shape: (num_points, data_dimension)
        self.data_points = data_points
        self.figures_dir = figures_dir
        self.palette = None
        self.labels = None

    @abstractmethod
    def run(self):
        pass

    def euclidean_distance(self, u: np.array, v: np.array) -> float:
        return np.sqrt(sum([(a-b)**2 for a, b in zip(u, v)]))

    def getDistanceMatrix(self) -> np.array:

        N_points, data_dim = self.data_points.shape
        distances = np.zeros((N_points, N_points))

        # measure distances between points
        for i, data_point_0 in enumerate(self.data_points):
            for j, data_point_1 in enumerate(self.data_points[(i+1):]):
                distance = self.euclidean_distance(data_point_0, data_point_1)
                distances[i, j+(i+1)] = distance
                distances[j+(i+1), i] = distance
        return distances

    def getPalette(self, N: int):
        if not self.palette or len(self.palette) != N:
            hex = list(range(10))+list('ABCDEF')
            self.palette = ['#{}'.format(''.join(list(np.random.choice(hex, 3)))) for _ in range(N)]
        return self.palette

    def paint(self, labels: np.array, iter: int, centroids=None, dim_support=2) -> None:
        # only support painting for a 2dim case
        N_points, data_dim = self.data_points.shape
        unique_labels = len(np.unique(labels))
        if data_dim > dim_support or not self.figures_dir:
            return

        # create data structure with points and labels
        # data points of shape: (num_points, data_dimension)
        # labels of shape: (num_points,)
        color = 'Color'

        def plot(d_np: np.array, to_append: np.array, ax=None, s=100, alpha=1.0):
            data = np.append(d_np, to_append[..., np.newaxis], axis=1)
            df_points = pd.DataFrame(data, columns=list(range(dim_support)) + [color])

            # save image of data_points and labels
            ax = sns.scatterplot(data=df_points,
                                 x=0,
                                 y=1,
                                 hue=color,
                                 palette=self.getPalette(unique_labels),
                                 ax=ax,
                                 s=s,
                                 alpha=alpha)

            return ax

        ax = plot(d_np=self.data_points, to_append=labels)
        if centroids is not None:
            ax = plot(d_np=centroids, to_append=np.array(list(range(unique_labels))), ax=ax, s=500, alpha=.5)

        plt.rcParams["font.weight"] = "bold"
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Epoch: ' + str(iter), fontsize=15, fontweight='bold')
        ax.get_legend().remove()
        out_file = os.path.join(self.figures_dir, str(iter))
        ax.get_figure().savefig(out_file, bbox_inches='tight')
        plt.clf()


    def makeGIF(self):
        figures = os.listdir(self.figures_dir)
        if not figures:
            return

        # https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
        import imageio
        images = []
        for i, _ in enumerate(figures):
            file_path = os.path.join(self.figures_dir, '{}.png'.format(i))
            images.append(imageio.imread(file_path))
        imageio.mimsave(os.path.join(self.figures_dir, 'vis.gif'), images, fps=0.75)