"""Compilation of tools to assess data for efficacy in machine learning models"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

auto_correlation = 0
partial_auto_correlation = 0
decompose = 0
stationary = 1

series = read_csv('humidityhourdiff.csv', header=0, squeeze=True, usecols=[1])

if(auto_correlation):
    plot_acf(series)
    plt.show()

if(partial_auto_correlation):
    plot_pacf(series)
    plt.show()

if(decompose):
    decomposed_series = seasonal_decompose(series, period=1)
    decomposed_series.plot()
    plt.show()

if(stationary):
    
    adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(series)
    print('ADF test statistic:', adf)
    print('ADF p-values:', pval)
    print('ADF number of lags used:', usedlag)
    print('ADF number of observations:', nobs)
    print('ADF critical values:', crit_vals)
    print('ADF best information criterion:', icbest)