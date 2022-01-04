# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:49:31 2021

@author: Henrique
"""

import pandas as pd
import ltspice as lt
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR, SVC
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


"""
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
"""
"""       
times = 1
df = pd.DataFrame(columns=['Res','V','I'])

x_list = []

while times <= 500:
    R = random.gauss(100, 25)
    x_list.append(R)
    f = open('ML_netlist.cir','w')
    f.write(f'ML\nV1 1 0 5\nR1 1 2 10\nR2 2 0 {R}\n.OP\n.PRINT V(2) I(R2)\n.END')
    f.close()

    #spice_path = find('XVIIx64.exe', 'D:\\')
    os.system('D:\\XVIIx64.exe -b ML_netlist.cir')

    #file = find('ML_netlist.raw', 'D:\\')
    l = lt.Ltspice('D:\\Spyder\\ML_netlist.raw')
    l.parse()
    
    volt = l.getData('V(2)')
    current = l.getData('I(R2)')
    data = {'Res' : R, 'V' : float(volt), 'I' : float(current)}
    df = df.append(data, ignore_index=True)
    
    print(times)    
    times +=1

data_csv = open('ltspice training data.csv','w')
for i in range(len(df)):
    data_csv.write(df.iloc[i,0] + ',' + df.iloc[i,1], + ',' + df.iloc[i,2])
data_csv.close()
dataset = pd.read_csv('ltspice training data.csv')
"""
df = pd.read_csv('ltspice training data.csv')
zero_train = []

t = 1

while t <= 500:
    zero_train.append(0)
    t += 1

#for time in times:
#    zero_array.apend(0)



zero_train = pd.DataFrame(zero_train)
zero_test = zero_train



x_train = pd.concat([df.iloc[:300,0], zero_train.iloc[:300]],axis=1)
x_test = pd.concat([df.iloc[300:,0], zero_test.iloc[300:]],axis=1)
y_train = df.iloc[:300,2]
y_test = df.iloc[300:,2]


svr_ml = SVR(kernel = 'poly',degree = 12)
svr_ml.fit(x_train,y_train)

y_pred = svr_ml.predict(x_test)
y_test = np.array(y_test)

#plt.plot(y_pred)
#plt.plot(y_test)
#plt.hist(df.iloc[:,'Res'], len(df.iloc[:,'Res']))
plt.show()
plt.scatter(y_test,y_pred)
plt.show()
plt.hist(y_test)
plt.hist(y_pred)
plt.show()

report = metrics.r2_score(y_test,y_pred)

diff = []
for i in range(len(y_test)):
    for j in range(len(y_pred)):
        if i == j:
            diff.append(abs(y_test[i] - y_pred[j]))

avg_error = (sum(diff)/sum(y_test))*100




