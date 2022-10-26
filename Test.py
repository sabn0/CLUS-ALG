
# import packages
import argparse
import numpy as np
from Algorithms.KMeans import KMeans
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-k', '--NumCentroids', default=2, type=int)
    parser.add_argument('-f', '--FiguresDir', default='Figures', type=str)
    parser.add_argument('-d', '--Data', required=True, help='path to data file, npy')
    args = parser.parse_args()

    # create figure folder
    if not os.path.isdir(args.FiguresDir):
        os.mkdir(args.FiguresDir)
    else:
        for file_name in os.listdir(args.FiguresDir):
            file_name = os.path.join(args.FiguresDir, file_name)
            os.remove(file_name)

    data_points = np.load(args.Data)
    KMeans(K=args.NumCentroids, data_points=data_points, figures_dir=args.FiguresDir).run()
