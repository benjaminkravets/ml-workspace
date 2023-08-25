import pandas as pd
import numpy as np

from statistics import mean
prox = [0,1,2,3,4,5,6,7,8,9]

print(mean(prox))

ranga = 10
weights = [(1/(ranga * (ranga / 2 - .5))) * x for x in range(ranga)]


prox = [prox[x] * weights[x] for x in prox]

print(sum(prox))
