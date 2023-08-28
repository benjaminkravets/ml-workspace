# single exponential smoothing
from pandas import read_csv
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
# prepare data
data = read_csv('datashop\ppmdaily.csv', header=0, usecols=[1]).values
# create class
import warnings
import matplotlib.pyplot as plt
  
# suppress warnings
warnings.filterwarnings('ignore')
mass = 1
masshistory = []
for i in range(len(data)):
    if i < 1000:
        continue
    model = ExponentialSmoothing(data[0:i])
    # fit model
    model_fit = model.fit(damping_trend=.2)
    
    # make prediction
    yhat = model_fit.predict()

    #print("Context:",data[i-5:i],"Next:",data[i])
    if(0):
        if yhat > 0:
            mass *= (1 + data[i])
        elif yhat < 0:
            mass *= 1 / (1 + data[i])
    if(1):
        if yhat > data[i-1]:
            mass *= (data[i] / data[i-1])
        elif yhat < data[i-1]:
            mass *= (data[i-1] / data[i])

        #print((data[i] / data[i-1]), (data[i-1] / data[i]))

    masshistory.append(mass[0])

    if i % 260 == 0:
        print(i,mass)

plt.plot(masshistory)
plt.show()