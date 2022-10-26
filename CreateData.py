
#import packages
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--SaveData', required=True, type=str, help='path to save data file')
    parser.add_argument('-n', '--NumPoints', default=100, type=int)
    parser.add_argument('-d', '--DimPoints', default=2, type=int)
    args = parser.parse_args()

    N, D = args.NumPoints, args.DimPoints
    data_points = np.random.rand(N, D)
    np.save(args.SaveData, data_points)