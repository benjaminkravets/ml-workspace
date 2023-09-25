
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

metric6 = []

def calculate_weighted_average_cv(category_float_pairs):
	"""Calculates the weighted average coefficient of variation (weighted average CV) for a group of category and float pairs.

	Args:
		category_float_pairs: A list of category and float pairs, where each pair is a tuple of (category, float value).

	Returns:
		The weighted average CV.
	"""

	# Create a DataFrame from the list of pairs.
	df = pd.DataFrame(category_float_pairs, columns=["category", "float"])

	# Calculate the mean float value for each category.
	mean_float_per_category = df.groupby("category")["float"].mean()

	# Calculate the standard deviation of the float values for each category.
	std_float_per_category = df.groupby("category")["float"].std()

	# Calculate the coefficient of variation (CV) for each category.
	cv_per_category = std_float_per_category / mean_float_per_category

	# Calculate the number of data points in each category.
	count_per_category = df.groupby("category")["float"].count()

	# Calculate the weighted average CV.
	weighted_average_cv = (cv_per_category * count_per_category).sum() / count_per_category.sum()

	return weighted_average_cv

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
		
				""" 
				metric0.append(math.log(prox[0] / prox[1]))
				metric1.append(math.log(prox[1] / prox[2]))
				metric2.append(math.log(prox[2] / prox[3]))
				metric3.append(math.log(prox[3] / prox[4]))
				metric4.append(math.log(prox[4] / prox[5]))
				metric5.append(math.log(prox[5] / prox[6]))
				 """
				
				#metric0.append(stdev(prox[0:5]))
				guage = ''

				for a, b, in itertools.combinations([two, three, four, five, ten, twenty, prox[0], hundred, twohundred], 2):
					#if(a > b):
					#    key += str(1)
					#else:
					#    key += str(0) 
					guage += str(a >= b)

				curtuple = (guage, diff)
				metric6.append(curtuple)
				#print(curtuple)

		for z, i in enumerate([metric0, metric1, metric2, metric3, metric4, metric5]):

				if len(i) == len(diffhistory):
						print("metric",z,statistics.correlation(diffhistory, i))

		for z, i in enumerate([metric6]):
				print(len(diffhistory), len(metric6))
				if len(i) == len(diffhistory):
						weighted_average_cv = calculate_weighted_average_cv(metric6)
						print(weighted_average_cv)


				
glower()