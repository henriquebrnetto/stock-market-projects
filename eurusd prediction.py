# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: henrique bucci rodrigues netto
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from numpy import *

sns.set()

dataset = pd.read_csv('HistoricalPrices.csv')
dataset = dataset.sort_index(ascending=False,ignore_index=True)

#Closing Price
y = dataset.iloc[:,-1].values
#Info
X = dataset.iloc[:,1:-1].values

#transforming into Data Frames
y = pd.DataFrame(y)
X = pd.DataFrame(X,columns=['Open','High','Low'])

#Simple Moving Average (SMA)
rolling_mean20 = y.rolling(window=20).mean()
rolling_mean50 = y.rolling(window=50).mean()

#Exponencial Moving Average (EMA)
exp20 = y.ewm(span=20, adjust=False).mean()
exp50 = y.ewm(span=50, adjust=False).mean()

y = np.array(y)
rolling_mean20 = np.array(rolling_mean20)
rolling_mean50 = np.array(rolling_mean50)

#RSL function
def RSL(close,maverage):
    RSL = close/maverage
    return RSL

RSL20_values = []
RSL50_values = []

for i in range(len(y)):
    res = RSL(y[i],rolling_mean20[i])
    RSL20_values.append(res)
    res2 = RSL(y[i],rolling_mean50[i])
    RSL50_values.append(res2)

RSL20_values = np.array(RSL20_values)
RSL50_values = np.array(RSL50_values)

def tr(high,low,close):
    tr=max(high,close)-min(low,close)
    return tr

TR_values = []

#Function to calculate ATR
for i in range(len(dataset)):
    if i > 0:
        trval = tr(dataset[' High'][i],dataset[' Low'][i],dataset[' Close'][i-1])
        TR_values.append(trval)

TR_values.insert(0,'nan')
TR_values = pd.DataFrame(TR_values)

#Defining ATR values
for i in range(len(TR_values)):
    ATR_values = TR_values.rolling(window=14).mean()

ATR_values = np.array(ATR_values)

#Proper DataFrame
so_df = dataset.iloc[:,2:].values
so_df = pd.DataFrame(so_df,columns=['High','Low','Close'])

#Stochastic Oscilator
for p in range(len(so_df)):
    so14_values = (so_df['Close'] - so_df['Low'].rolling(window=14).min())/(so_df['High'].rolling(window=14).max() - so_df['Low'].rolling(window=14).min())

#Stochastic Oscilator 3
for p in range(len(so_df)):
    so3_values = so14_values.rolling(window=3).mean()

#Hurst Function
z = y[-1000:]
lags = range(2, 20)
tau = [sqrt(std(subtract(z[lag:], z[:-lag]))) for lag in lags]
plt.plot(log(lags), log(tau)); plt.show()
m = polyfit(log(lags), log(tau),1)
hurst = m[0]*2
print('hurst = ',hurst)

#Transforming the parameters into DataFrame
RSL20_values = pd.DataFrame(RSL20_values)
RSL50_values = pd.DataFrame(RSL50_values)
ATR_values = pd.DataFrame(ATR_values)
so14_values = pd.DataFrame(so14_values)
so3_values = pd.DataFrame(so3_values)

#Creating Features Dataset
data = pd.concat([RSL20_values,ATR_values,so14_values,so3_values],axis=1)
feats = pd.DataFrame(data)
feats = feats.dropna()
feats = np.array(feats)

final_result = []

#Result to be Predicted
for i in range(len(y)):
    if i+1 == 3705:
        final_result.append('nan')
        break
    if i>=19:
        if y[i+1]>y[i]:
            final_result.append('increase')
        if y[i+1]<y[i]:
            final_result.append('decrease')
        if y[i+1]==y[i]:
            if y[i+2]>y[i]:
                final_result.append('increase')
            if y[i+2]<y[i]:
                final_result.append('decrease')

#Transform String to Int
final_result = np.array(final_result)
le = LabelEncoder()
final_result = le.fit_transform(final_result)

#Final Features Data
feats = pd.DataFrame(feats, columns=['RSL20_values','ATR_values','so14_values','so3_values'])

#Final Prediction Data
final_result= pd.DataFrame(final_result, columns=['+/-'])

#splitting train and test sets
X_train = feats.iloc[:-941,:].values
X_test = feats.iloc[-941:-1,:].values
y_train = final_result.iloc[:-941].values
y_test = final_result.iloc[-941:-1].values

y_train = y_train.reshape(2745,)
y_test = y_test.reshape(940,)

#The Final Dataset with the Information Used
final_data = pd.concat([feats, final_result],axis=1)

#scaling the training and testing data
mms = MinMaxScaler()
X_train,X_test = mms.fit_transform(X_train), mms.fit_transform(X_test)

#KNN
knn = KNeighborsClassifier(n_neighbors=12, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
report = metrics.classification_report(y_test,y_pred)

lastpred = final_data.iloc[-1,:-1].values
lastpred = lastpred.reshape(1,-1)
print("[1] = increased value","\n[0] = decreased value")
print("Prediction = ",knn.predict(lastpred))