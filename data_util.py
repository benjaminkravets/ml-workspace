#import pandas as pd 
import numpy as np 

def data_gen(start_slider, end_slider):

    master = np.genfromtxt("Book1.csv",delimiter=',',skip_header=1)

    X_master, y_master = master[:, :-1], master[:, -1]

    X_test, y_test = np.copy(X_master), np.copy(y_master)

    start, stop = int(start_slider * len(X_master)), int(end_slider * len(X_master)), 

    X_train, y_train = np.copy(X_master[start:stop]), np.copy(y_master[start:stop])

    feature_count = X_train.shape[-1]
    
    return X_train, y_train, X_test, y_test, feature_count


if __name__ == "__main__":

    X_train, y_train, X_test, y_test, feature_count = data_gen(0,.5)

