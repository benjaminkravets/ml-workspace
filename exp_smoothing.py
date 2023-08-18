# single exponential smoothing
from pandas import read_csv

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# prepare data
data = read_csv('datashop/humidityhourdiff.csv', header=0, usecols=[1])
# create class
model = SimpleExpSmoothing(data)
# fit model
model_fit = model.fit()
# make prediction
yhat = model_fit.predict()