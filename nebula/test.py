import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras import layers
import time

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(float)


#root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

#x_train, y_train = readucr("FordA_TRAIN.tsv")
#x_test, y_test = readucr("FordA_TEST.tsv")

x_train, y_train = readucr("test.txt")
x_test, y_test = readucr("test.txt")

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


n_classes = len(np.unique(y_train))

z = 1000




def joe():
    global z
    reconstructed = keras.models.load_model("mostrecent")
    unit = 0
    history = []

    
    #score = reconstructed.evaluate(x_train, y_train, verbose = 2) 
    #print('Test loss:', score[0]) 
    #print('Test accuracy:', score[1])


    prediction = reconstructed.predict(x_train[0:1], verbose = 0)

    #print(" ")
    #print("x vals: " + str(x_train[0:1]))
    #print(" y vals: " + str(prediction))
    #print("actual: " + str(y_train[0]))


    for i in range(6100):
        
        prediction = reconstructed.predict(x_train[i:i+1], verbose = 0)
        if(0):
            print(" ")
            print("x vals: " + str(x_train[i:i+1]))
            print("x vals shape: " + str(x_train[i:i+1].shape))
            print(" y vals: " + str(prediction))
            print("actual: " + str(y_train[i]))
            time.sleep(1)
        if(prediction > .02):
            z = z * (1 + y_train[i] / 100 * 1) - 0
            

        if(prediction < -.02):
            z = z * (1 - y_train[i] / 100 * 1) - 0
            pass

        if(i % 20 == 0 and 1):
            print(str(unit) + " " + str(z))
            unit += 1
        history.append(z)

    plt.plot(history)
    plt.show()
            

def clair():
    return 0


joe()