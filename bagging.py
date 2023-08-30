# bagging ensemble for making predictions for regression
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# define dataset
#X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=5)
if(1):
    data = read_csv("datashop/spydailydiff.csv")
    data = data['Open'].tolist()
    #print(len(data))
    #data = preprocessing.StandardScaler().fit_transform(data)
    ccdata = data
    l = 2
    data = data[l:len(data)]
    #data2 = data2[l:len(data2)]
    ccdata = ccdata[l:len(ccdata)]

    x = []
    y = []
    look_back = 20

    for i in range(int((len(data)-look_back) * .3)):
        #xvals = np.append(data[i:i+look_back], mean(data[i:i+look_back]))
        #xvals = np.atleast_2d(xvals)
        #print(xvals)
        xvals = data[i:i+look_back]
        #print(xvals)
        xvals.append(np.std(data[i:i+look_back]))

        #print(xvals)
        #print()
        x.append(xvals)
        y.append(ccdata[i+look_back])

    i = 0

    X = np.array(x)
    y = np.array(y)
#print(type(data))
# define the model
j = 100
model = BaggingRegressor(n_estimators=10, max_features=.5)
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
#row = [[0.88950817,-0.93540416,0.08392824,0.26438806,-0.52828711,-1.21102238,-0.4499934,1.47392391,-0.19737726,-0.22252503,0.02307668,0.26953276,0.03572757,-0.51606983,-0.39937452,1.8121736,-0.00775917,-0.02514283,-0.76089365,1.58692212]]
#yhat = model.predict(row)
#print('Prediction: %d' % yhat[0])

mass = 1
masshistory = []

for i in range(len(data)-look_back):
    #xvals = np.append(data[i:i+look_back], mean(data[i:i+look_back]))

    xvals = [data[i:i+look_back]]
    #print(xvals, type(xvals))
    xvals[0].append(np.std(data[i:i+look_back]))

    #print(xvals, type(xvals))
    
    obs = ccdata[i+look_back]
    yhat = model.predict(xvals)
    ycur = ccdata[i+look_back-1]
    if(1):
        if yhat > 0:
            mass *= (1 + obs)
        elif yhat < 0:
            mass *= 1 / (1 + obs)
    if(0):
        if yhat > ycur:
            mass *= (obs / ycur)
        elif yhat < ycur:
            mass *= (ycur / obs)


    masshistory.append(mass)

print(mass)

plt.plot(masshistory)
plt.show()