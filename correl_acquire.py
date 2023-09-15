
import csv
from statistics import mean
from statistics import stdev
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime
import time
import pickle
from dateutil.parser import parse
import time
import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Thread
import itertools
import time
import multiprocessing
import concurrent.futures
import math
import sys
import statistics

targetcsv = "datashop/humiditydaily.csv"
pl = 200

diffhistory = []

metric0 = []
metric1 = []
metric2 = []
metric3 = []
metric4 = []
metric5 = []

def glower():
    index = 0
    global curdate
    global nextopen
    prox = [1 for x in range(pl)]
    file = open(targetcsv, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()


    for z, row in enumerate(data):
        if(z == 0):
            continue
        prox.insert(0,float(row[1]))
        nextopen = row[6]
        
        curdate = parse(row[0])
        del prox[pl]
        
        index = index + 1

        if(index < 201):
            continue

        fourhundred = mean(prox[0:400])
        #threehundred = mean(prox[0:300])
        twohundred = mean(prox[0:200])
        hundredfifty = mean(prox[0:200])
        hundred = mean(prox[0:100])
        #eighty = mean(prox[0:80])
        sixty = mean(prox[0:60])
        fifty = mean(prox[0:50])
        forty = mean(prox[0:40])
        thirty = mean(prox[0:30])
        #twentysix = mean(prox[0:26])
        twenty = mean(prox[0:20])
        fifteen = mean(prox[0:15])
        ten = mean(prox[0:10])
        #nine = mean(prox[0:9])
        five = mean(prox[0:5])
        four = mean(prox[0:4])
        three = mean(prox[0:3])
        two = mean(prox[0:2])

        diff = float(nextopen) / prox[0]
        diffhistory.append(diff)
    

        metric0.append(prox[0] / prox[1])
        metric1.append(prox[1] / prox[2])
        metric2.append(prox[2] / prox[3])
        metric3.append(prox[3] / prox[4])
        metric4.append(prox[4] / prox[5])
        metric5.append(prox[5] / prox[6])



    for z, i in enumerate([metric0, metric1, metric2, metric3, metric4, metric5]):

        if len(i) == len(diffhistory):
            print("metric",z,statistics.correlation(diffhistory, i))


        
glower()