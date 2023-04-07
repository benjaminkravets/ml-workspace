import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras import layers
import time
import csv
import numpy







def joe():
    z = 100
    model = keras.models.load_model("mostrecent")
    # choose a number of time steps
    n_steps = 100
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    #reconstructed = keras.models.load_model("mostrecent")
    file = open("test.txt", "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()

    data = [val for sublist in data for val in sublist]

    data = [float(i) for i in data]
    index = 0
    period = n_steps
    for x in data:
        twentyday = data[index:index+period]
        index += 1
        if len(twentyday) == period:

            #print(twentyday)

            x_input = numpy.array(twentyday)
            
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            if(yhat > 0):
                #print("Predicted: " + str(yhat) + " Actual: " + str(data[index+21]))
                z = z * (1 + data[index + period + 1] / 100 * 1) - 0
            if(yhat < 0):
                #print("Predicted: " + str(yhat) + " Actual: " + str(data[index+21]))
                z = z * (1 - data[index + period + 1] / 100 * 1) - 0
            print(z)
            





joe()