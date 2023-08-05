from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose




dataframe = read_csv('humidity.csv', usecols=[1], engine='python')
dataset = dataframe.values
series = dataset.astype('float32')

for i,x in enumerate(series):
    series[i] = series[i] / series[i-1]

result = seasonal_decompose(series, model='multiplicative', period=1)

result.plot()
pyplot.show()

print(result.seasonal[0:1000])