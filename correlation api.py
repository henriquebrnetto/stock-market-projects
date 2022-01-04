# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:41:16 2021

@author: Henrique
"""

import requests
import json
import matplotlib.pyplot as plt
import pandas as pd

url = "https://api.twelvedata.com/correl"

parameters = {"symbol" : ["TSLA","VWAGY"], 'interval':'1day','outputsize':'5000', "format" : "json", 'time_period' : '10' ,'apikey':'764531eaee034835bb31c961cb11bf1c'}

response = requests.get(url, params=parameters).json()

df = pd.DataFrame(columns = ['date','correlation'])

for correlation in response['values']:
    date = correlation['datetime']
    corr = correlation['correl']
    df = df.append({'date' : date, 'correlation' : corr}, ignore_index = True)
    

    
    