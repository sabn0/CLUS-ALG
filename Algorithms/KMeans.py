
# import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import os


class KMeans:
    def __init__(self,
                 data_points: np.array,
                 K: int,
                 figures_dir=None,
                 max_iter=10,
                 ):

        assert K > 0, "invalid number of centroids provided: {}".format(K)

        # data points of shape: (num_points, data_dimension)
        self.data_points = data_points
        self.max_iter = max_iter
        self.K = K
        self.figures_dir = figures_dir
        self.palette = None
        self.labels = None

    def getPalette(self):
        if not self.palette:
            hex = list(range(10))+list('ABCDEF')
            self.palette = ['#{}'.format(''.join(list(np.random.choice(hex, 3)))) for _ in range(self.K)]
        return self.palette

    def paint(self, labels: np.array, centroids: np.array, iter: int, dim_support=2) -> None:

        # only support painting for a 2dim case
        N_points, data_dim = self.data_points.shape
        if data_dim > dim_support or not self.figures_dir:
            return

        # create data structure with points and labels
        # data points of shape: (num_points, data_dimension)
        # labels of shape: (num_points,)
        color = 'Color'

        def plot(d_np: np.array, to_append: np.array, ax=None, s=100, alpha=1.0):

            data = np.append(d_np, to_append[..., np.newaxis], axis=1)
            df_points = pd.DataFrame(data, columns=list(range(dim_support))+[color])

            # save image of data_points and labels
            ax = sns.scatterplot(data=df_points,
                                 x=0,
                                 y=1,
                                 hue=color,
                                 palette=self.getPalette(),
                                 ax=ax,
                                 s=s,
                                 alpha=alpha)

            return ax

        ax = plot(d_np=self.data_points, to_append=labels)
        ax = plot(d_np=centroids, to_append=np.array(list(range(self.K))), ax=ax, s=500, alpha=.5)
        plt.rcParams["font.weight"] = "bold"
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('Epoch: '+ str(iter), fontsize=15, fontweight='bold')
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


    def run(self):

        # expectation - maximization implementation for n'dim data
        N_points, data_dim = self.data_points.shape

        # create initial centroids
        centroids = np.random.rand(self.K, data_dim)
        labels = np.random.choice(self.K, N_points)

        for iter in range(self.max_iter):

            # paint process
            self.paint(labels, centroids, iter=iter)

            # expectation step :
            # calculate (euc) distance from each data_point to every centroid
            # and find the closest centroid for the data point
            for i, data_point in enumerate(self.data_points):
                distances = []
                for centroid in centroids:
                    distances += [np.sqrt(sum([(a-b)**2 for a, b in zip(data_point, centroid)]))]

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

