import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys

import pmdarima as pm
from pandas import read_csv
from pmdarima.datasets.stocks import load_msft

df = load_msft()
df.head()

train_len = int(df.shape[0] * 0.8)
train_data, test_data = df[:train_len], df[train_len:]

y_train = train_data['Open'].values
y_test = test_data['Open'].values

series = read_csv('datashop/ppmdailydiff.csv', header=0, usecols=[1]).values
y_train = series[0:3000]
y_test = series[3000:len(series)]


print(f"{train_len} train samples")
print(f"{df.shape[0] - train_len} test samples")

from pmdarima.arima import ndiffs

kpss_diffs = ndiffs(y_train, alpha=0.01, test='kpss', max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.01, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")

auto = pm.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,
                     suppress_warnings=True, error_action="ignore",
                     max_order=None, trace=True)

from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

model = auto

def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

forecasts = []
confidence_intervals = []

z = 0
mass = 1
for i,new_ob in enumerate(y_test):
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)
    #print(fc, conf)
    z += 1
    tol = .000
    if(0):
        if fc > y_test[i] + tol:
            mass *= ((y_test[i+1] / y_test[i] - 1) * 1 + 1)
        elif fc < y_test[i] - tol:
            mass *= ((y_test[i] / y_test[i+1] - 1) * 1 + 1)
        print(mass)

    if(1):
        if fc > 0:
            mass *= (1 + y_test[i+1])
        elif fc < 0:
            mass *= 1 / (1 + y_test[i+1])
        print(mass)
    # Updates the existing model with a small number of MLE steps
    model.update(new_ob)
    
print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
print(f"SMAPE: {smape(y_test, forecasts)}")
