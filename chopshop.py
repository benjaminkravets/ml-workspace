# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
import joblib


# normalize the dataset
scaler = joblib.load('scalar.save')
#model = keras.models.load_model("models/model.keras")

sample = np.ndarray([[1], [1], [1], [1], [1]]).astype('float32')

sample = scaler.transform(sample)

