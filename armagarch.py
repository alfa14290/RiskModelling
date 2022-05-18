import ticker as readData
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pmd
from arch import arch_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def stationary(res):
    res.dropna(inplace=True)
    X= res['return'].values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if result[0] < result[4]["5%"]:
        print ("Reject Ho - Time Series is Stationary")
    else:
        print ("Failed to Reject Ho - Time Series is Non-Stationary")

def arimamodel(timeseriesarray):
    autoarima_model = pmd.auto_arima(timeseriesarray, 
                              start_p=1, 
                              start_q=1,
                              test="adf",
                              trace=True)
    p,d,q = autoarima_model.order
    
    model = ARIMA(timeseriesarray, order=(p,d,q))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()
    skewt_gm = arch_model(residuals, p=1, q=1, vol='GARCH') 
    skewt_result = skewt_gm.fit()
    print(skewt_result.params)
    skewt_volatility = skewt_result.conditional_volatility
    skewt_resid = skewt_result.resid/skewt_volatility
    # Plot model fitting results
    plt.figure(figsize=(12,6))
    plt.plot(skewt_volatility, color = 'black', label = 'Skewed-t Volatility')
    plt.plot(res['return'], color = 'grey', label = 'Daily Returns', alpha = 0.4)
    plt.legend(loc = 'upper right', frameon=False)
    plt.show()
    nu = skewt_result.params[3]

# t distribution skew (lambda)
   # lam = skewt_result.params[4]

# Forecast from 2018-01-01 onward 
    #start_loc = df[df['Date']=='02-01-2018'].index
    #print(start_loc)
    garch_forecast = skewt_result.forecast(start='2018-01-01')
    ARMA_forecast = model_fit.predict(start='2018-01-02')
    print(ARMA_forecast.head())

# Forecast mean and variance 
    mean_forecast = ARMA_forecast.values.reshape(-1,1)
    print(mean_forecast)
    print(mean_forecast.shape)
    variance_forecast = garch_forecast.variance['2018-01-01':]

    # Parametric quantile, 2nd argument of .ppf must be in an array form
    q_parametric = skewt_gm.distribution.ppf(0.05)
    print('5% parametric quantile: ', q_parametric)

    # Parametric VaR
    VaR_parametric = mean_forecast + np.sqrt(variance_forecast).values * q_parametric

    
    VaR_parametric = pd.DataFrame(VaR_parametric, columns = ['5%'], index = variance_forecast.index)

    # Empirical quantilen
    q_empirical = skewt_resid.quantile(0.05)
    print('5% empirical quantile: ', q_empirical)

    # Emperical VaR
    VaR_empirical = mean_forecast + np.sqrt(variance_forecast).values * q_empirical
    VaR_empirical = pd.DataFrame(VaR_empirical, columns = ['5%'], index = variance_forecast.index)
    print(VaR_empirical.index.size)
    # Plot
    plt.figure(figsize=(12,7))
    plt.plot(VaR_parametric, color = 'salmon', label = '5% Parametric VaR', alpha=0.7)
    plt.plot(VaR_empirical, color = 'gold', label = '5% Empirical VaR', alpha=0.7)

    df= res.reset_index(inplace=True)
    df = res.rename(columns = {'index':'Date_fcx'})
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




res = readData.read("PTY")

stationary(res)
arima_model = arimamodel(res['return'])
#arima_model.summary()

