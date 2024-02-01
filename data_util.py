#import pandas as pd 
import numpy as np 

def data_gen(start_slider, end_slider):

    master = np.genfromtxt("Book1.csv",delimiter=',',skip_header=1)

    X_master, y_master = master[:, :-1], master[:, -1]

    X_test, y_test = np.copy(X_master), np.copy(y_master)

    start_training_index, stop_training_index = int(start_slider * len(X_master)), int(end_slider * len(X_master)), 

    print(start_training_index, stop_training_index)
    


if __name__ == "__main__":
    data_gen(0,1)
