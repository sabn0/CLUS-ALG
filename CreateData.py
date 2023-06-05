
#import packages
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--SaveData', required=True, type=str, help='path to save data file')
    parser.add_argument('-a', '--Algorithm', required=True, type=str, help='name of algorithm')
    parser.add_argument('-n', '--NumPoints', default=200, type=int)
    parser.add_argument('-d', '--DimPoints', default=2, type=int)
    args = parser.parse_args()

    N, D = args.NumPoints, args.DimPoints

    if args.Algorithm == 'KMeans':
        # for kmeans
        data_points = np.random.rand(N, D)
    elif args.Algorithm == 'DBScan':
        # for dbscan
        lower, upper = 0.4, 0.6
        data_points = np.random.uniform(lower,upper,[N//5,D])
        while True:
            outer_point = np.random.uniform(0, 1, [1, D])
            if any(d > upper+0.2 or d < lower-0.2 for d in outer_point[0]):
                data_points = np.concatenate((data_points, outer_point))
            if data_points.shape[0] >= N:
                break
    else:
        raise ValueError("unrecognized alogrithm {}".format(args.Algorithm))

    
    np.save(args.SaveData, data_points)