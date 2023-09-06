"""Compilation of tools to assess data for efficacy in machine learning models"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
import sys
from pandas import DataFrame
each = 1
auto_correlation = 0 or each
partial_auto_correlation = 0 or each
decompose = 1 or each
stationary = 0 or each
lag_plot_l = 0 or each

series = read_csv('datashop/humiditydailydiff.csv', header=0, usecols=[1])

if(auto_correlation):
    plot_acf(series)
    plt.show()

if(partial_auto_correlation):
    plot_pacf(series)
    plt.show()

if(decompose):

    series = series.values
    decomposed_series = seasonal_decompose(series, period=2, two_sided=False).resid
    plt.plot(decomposed_series)
    plt.show()

if(stationary):
    
    adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(series)
    print('ADF test statistic:', adf)
    print('ADF p-values:', pval)
    print('ADF number of lags used:', usedlag)
    print('ADF number of observations:', nobs)
    print('ADF critical values:', crit_vals)
    print('ADF best information criterion:', icbest)

if(lag_plot_l):
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    plt.title('MSFT Autocorrelation plot')

    # The axis coordinates for the plots
    ax_idcs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1)
    ]

    for lag, ax_coords in enumerate(ax_idcs, 1):
        ax_row, ax_col = ax_coords
        axis = axes[ax_row][ax_col]
        lag_plot(DataFrame(series), lag=lag, ax=axis)
        axis.set_title(f"Lag={lag}")

    plt.show()