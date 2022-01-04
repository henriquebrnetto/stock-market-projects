# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:24:24 2021

@author: Henrique
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#Hurst Exponent function
def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

#RSL function
def RSL(close,maverage):
    RSL = close/maverage
    return RSL


url = 'https://api.twelvedata.com/time_series?apikey=764531eaee034835bb31c961cb11bf1c&interval=1day&symbol=EUR/USD&type=index&outputsize=5000&dp=10&previous_close=true&format=JSON'

response = requests.get(url).json()

df = pd.DataFrame(columns = ['date','close'])

for line in response['values']:
    date = line['datetime']
    close = float(line['close'])
    df = df.append({'date' : date, 'close' : close}, ignore_index=True)
    
y = df.iloc[:,-1]
    
"""
df = pd.read_csv('eurusd data api.csv')
y = df.iloc[:,1].values
"""

#Hurst Exponent
hurst_data = df.iloc[-500:,-1].values
hurst = get_hurst_exponent(hurst_data)

#Simple Moving Average (SMA)
rolling_mean20 = y.rolling(window=20).mean()
rolling_mean50 = y.rolling(window=50).mean()

#Exponencial Moving Average (EMA)
exp20 = y.ewm(span=20, adjust=False).mean()
exp50 = y.ewm(span=50, adjust=False).mean()


RSL20_values = []
RSL50_values = []

for i in range(len(y)):
    res = RSL(y[i],rolling_mean20[i])
    RSL20_values.append(res)
    res2 = RSL(y[i],rolling_mean50[i])
    RSL50_values.append(res2)



























