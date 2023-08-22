# single exponential smoothing
from pandas import read_csv

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
# prepare data
data = read_csv('datashop/spydailydiff.csv', header=0, usecols=[1]).values
# create class

mass = 1
for i in range(len(data)):
    if i < 200:
        continue
    model = ExponentialSmoothing(data[i-20:i])
    # fit model
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict()

    #print("Context:",data[i-5:i],"Next:",data[i])

    if yhat > 0:
        mass *= (1 + data[i])
    if yhat < 0:
        mass *= 1 / (1 + data[i])
    print(mass)
