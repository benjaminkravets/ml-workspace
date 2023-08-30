# evaluate adaboost ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import AdaBoostRegressor
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)
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

    for i in range(int((len(data)-look_back) * 1)):
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
# define the model
model = AdaBoostRegressor(n_estimators= 100, learning_rate=.1)
model.fit(X, y)
# evaluate the model
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
#print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

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