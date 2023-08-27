import pandas as pd
import numpy as np

from statistics import mean
prox = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
prox = [3.2,5,4,7,10,11]
#print(mean(prox))

ranga = 5
#weights = [(x/(ranga * (ranga / 2 - .5))) for x in reversed(range(ranga))]
weights = [(2 * x) / ((ranga ** 2 - ranga) / 1) for x in reversed(range(ranga))]
prox2 = [prox[x] * weights[x] for x in range(ranga)]
weightedaverage = sum(prox2)
print(prox2)
print(weights, sum(weights))

print(weightedaverage)
