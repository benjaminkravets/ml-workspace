import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

series = read_csv('humidityhourdiff.csv', header=0, squeeze=True, usecols=[1])


model = pm.auto_arima(series, seasonal=True, m=52)
