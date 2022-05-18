import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import ticker as readData

def var_historic(r, year,level=1):
    var_hist = []

    for x in range(0,3):
        r_rolling = []
        for z in range(0,len(r)):
            if year[z]==(2017+x):
                r_rolling.append(r[z])

        var = np.percentile(r_rolling, level) 
        var_hist.append(var) 
       
        r_test = []
        for t in range(0,len(r)):
            if year[t]==(2018+x):
                r_test.append(r[t])
        print(len(r_test))
        print((sum(r_test < var)))
        
    
    return var_hist

def var_montecarlo(r, year ,level=1):
    var_mc = []
    for x in range(0,3):
        r_rolling = []
        for z in range(0,len(r)):
            if year[z]==(2018+x):
                r_rolling.append(r[z])
        
        r_mean = np.mean(r_rolling)
        std = np.std(r_rolling)
        Z = np.random.normal(r_mean, std, 1000000)
        var = np.percentile(Z, level) 
        var_mc.append(var) 
        
        print(len(r_rolling))
        print((sum(r_rolling < var)))
    
    return var_mc
res = readData.read('PTY')
PTY = res['return']
year_PTY = []
df= res.reset_index(inplace=True)
df = res.rename(columns = {'index':'Date'})
date_PTY = df["Date"]  


for y in range(0,len(date_PTY)):
    d = date_PTY[y]
    year_PTY.append(d.year)
VaR_hist_PTY = var_historic(PTY,year_PTY)
print(VaR_hist_PTY)

VaR_mc_PTY = var_montecarlo(PTY,year_PTY)

print(VaR_mc_PTY)


