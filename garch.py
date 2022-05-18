import pandas as pd
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model
import seaborn
from arch.unitroot import ADF
from statsmodels.tsa.stattools import adfuller
import numpy as np
from sympy import re
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox   
from sklearn.metrics import mean_absolute_error, mean_squared_error  
import ticker as readData
from datetime import datetime, timedelta

#### Global plotting style
seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 6))
plt.rc("savefig", dpi=90)
plt.rc("font", family="sans-serif")
plt.rc("font", size=14)     


def garch(res):
    res.dropna(inplace=True)
    garch = arch_model(res['return'], mean='zero', vol='GARCH', p=1, o=0, q=1)\
                 .fit(disp='off')
    print(garch.summary())
    #plt.plot(garch.forecast(start = res[-252:].index[0]).variance.iloc[-len(res[-252:].index):], label = 'fcx GARCH(1,1)', linewidth = .75)
    garch.plot()
    plt.show()
    gm_forecast = garch.forecast(horizon = 5)
    # Print the forecast variance
    print(gm_forecast.variance[-1:])
    gm_std_resid = garch.resid / garch.conditional_volatility
    print(gm_std_resid)
    ax = gm_std_resid.hist(color='salmon', bins=40)
    ax.set(title='Distribution of Standardized Residuals')
    plt.show()
    skewt_gm = arch_model(res['return'], p=1, q=1, mean='zero', vol='GARCH', dist='skewt') 
    skewt_result = skewt_gm.fit()
    print(skewt_result.summary())
    normal_volatility = garch.conditional_volatility
    skewt_volatility = skewt_result.conditional_volatility
    skewt_resid = skewt_result.resid/skewt_volatility
    def goodness_of_fit():
        #global df
        model_names = ['normal', 'skewt']
        models = [garch, skewt_result]
        likelihood = [model.loglikelihood for model in models]
        aic = [model.aic for model in models]
        bic = [model.bic for model in models]
        dict = {'model':model_names, 'log likelihood':likelihood, 'aic':aic,'bic':bic}
        df = pd.DataFrame(dict).set_index('model')
        return df

    goodness_of_fit()
    

    # Plot model fitting results
    plt.figure(figsize=(12,6))
    plt.plot(skewt_volatility, color = 'black', label = 'Skewed-t Volatility')
    plt.plot(normal_volatility, color = 'turquoise', label = 'Normal Volatility')
    plt.plot(res['return'], color = 'grey', label = 'Daily Returns', alpha = 0.4)
    plt.legend(loc = 'upper right', frameon=False)
    plt.show()

    plot_acf(skewt_resid, alpha=0.05)
    plt.show()


    daily_volatility = res['return'].std()
    print('Daily volatility: ', '{:.2f}%'.format(daily_volatility))

    monthly_trade_days = 21
    monthly_volatility = np.sqrt(monthly_trade_days) * daily_volatility
    print('Monthly volatility: ', '{:.2f}%'.format(monthly_volatility))

    yearly_trade_days = 252
    yearly_volatility = np.sqrt(yearly_trade_days) * daily_volatility
    print('Yearly volatility: ', '{:.2f}%'.format(yearly_volatility))

  

    ### reduce start loc and end loc for variance forcast as there are 3 for loops
    forecasts={}
    df= res.reset_index(inplace=True)
    df = res.rename(columns = {'index':'Date_fcx'})
    print(df.tail())
    start = df[df['Date']=='10-10-2019'].index[0]
    end = start + 60 ## fix the window to 60 days

    

    for i in range(30):
       skewt_gm = arch_model(df['return'], p=1, q=1, mean='zero', vol='GARCH', dist='skewt') 
       garch_fixed_rolling_result = skewt_gm.fit(first_obs = i + start, 
                                           last_obs = i + end, 
                                              update_freq = 5,
                                           disp='off')
       temp_result = garch_fixed_rolling_result.forecast(horizon = 1).variance
       fcast = temp_result.iloc[i + end]
       forecasts[fcast.name] = fcast
       forecast_var_fixed = pd.DataFrame(forecasts).T
    print(forecast_var_fixed.head())
    plt.figure(figsize=(10,6))
    plt.plot(forecast_var_fixed, color = 'lightcoral', label='Variance')
    plt.plot(df.iloc[start+60:end+30]['return'], color = 'gray', label='Return')
    plt.ylabel('%')
    plt.legend(frameon=False)
    plt.title('fcx Forecast Variance vs. Returns (Fixed Window)')  
    plt.show()
    print(garch_fixed_rolling_result.params) 

  ###actual volitiliy from garch Variance from model
    actual_var = skewt_result.conditional_volatility ** 2 
    actual_var = actual_var['2018-01-07':'2018-02-19']
    print(forecast_var_fixed)
    actual_var, forecast_var_fixed = np.array(actual_var), np.array(forecast_var_fixed["h.1"])
    print(actual_var)
    print(forecast_var_fixed)
    def evaluate(observation, forecast): 

        # MAE
        mae = mean_absolute_error(observation, forecast)
        print('Mean Absolute Error (MAE): {:.3g}'.format(mae))
   
        # MSE
        mse = mean_squared_error(observation, forecast)
        print('Mean Squared Error (MSE): {:.3g}'.format(mse))

        
        mape = np.mean(np.abs((observation - forecast) / observation)) * 100
        print('Mean Absolute Percentage Error (MAPE): {:.3g}'.format(mape))
    
        return mae, mse, mape

# Backtest model with MAE, MSE
    #evaluate(actual_var, forecast_var_fixed)

    nu = skewt_result.params[3]

# t distribution skew (lambda)
    lam = skewt_result.params[4]

# Forecast from 2018-01-01 onward 
    #start_loc = df[df['Date']=='02-01-2018'].index
    #print(start_loc)
    garch_forecast = skewt_result.forecast(start='2018-01-01')

# Forecast mean and variance 
    mean_forecast = garch_forecast.mean['2018-01-01':]
    variance_forecast = garch_forecast.variance['2018-01-01':]

    # Parametric quantile, 2nd argument of .ppf must be in an array form
    q_parametric = skewt_gm.distribution.ppf(0.05, [nu, lam])
    print('5% parametric quantile: ', q_parametric)

    # Parametric VaR
    VaR_parametric = mean_forecast.values + np.sqrt(variance_forecast).values * q_parametric
    VaR_parametric = pd.DataFrame(VaR_parametric, columns = ['5%'], index = variance_forecast.index)

    # Empirical quantile
    q_empirical = skewt_resid.quantile(0.05)
    print('5% empirical quantile: ', q_empirical)

    # Emperical VaR
    VaR_empirical = mean_forecast.values + np.sqrt(variance_forecast).values * q_empirical
    VaR_empirical = pd.DataFrame(VaR_empirical, columns = ['5%'], index = variance_forecast.index)
    print(VaR_empirical.index.size)
    # Plot
    plt.figure(figsize=(12,7))
    plt.plot(VaR_parametric, color = 'salmon', label = '5% Parametric VaR', alpha=0.7)
    plt.plot(VaR_empirical, color = 'gold', label = '5% Empirical VaR', alpha=0.7)
    
    df['date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    filtered_df = df.loc[(df['date'] >= '2018-01-01')]
    print(filtered_df.index.size)
    filtered_df['Date'] = pd.to_datetime(df['Date'])
    filtered_df.set_index('Date', inplace=True)
    print(filtered_df['return'])
    print(VaR_empirical['5%'])
    t = np.where(filtered_df['return'] < VaR_empirical['5%'])
    print([len(e) for e in t])
    colors = np.where(filtered_df['return'] < VaR_empirical['5%'],'red','turquoise') # where return<VaR, point is dark red
    plt.scatter(variance_forecast.index, filtered_df['return'], color = colors, label = 'FCX Returns')
    plt.legend(loc = 'upper right', frameon=False)
    plt.ylabel('%')
    plt.title('FCX VaR')
    plt.show()
    

res = readData.read("FCX")
#stationary(res)

garch(res)
#rollingwindow(res)
#modelFiteness(result1, result2)
