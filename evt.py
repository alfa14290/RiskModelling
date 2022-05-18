import matplotlib.pyplot as plt
import py
import ticker as readData
import pandas as pd
from arch import arch_model
from pyextremes import EVA 
from pyextremes import get_return_periods
import scipy.stats as stats
import pylab 
from pyextremes import plot_mean_residual_life
from pyextremes import plot_parameter_stability
from pyextremes import plot_return_value_stability
from pyextremes import plot_threshold_stability
from pyextremes import get_extremes
import numpy as np
import pot



def evtgarch(res):
    res.dropna(inplace=True)
    skewt_gm = arch_model(res['return'], p=1, q=1, mean='zero', vol='GARCH', dist='skewt') 
    skewt_result = skewt_gm.fit()
    print(skewt_result.summary())
    skewt_result.plot()
    plt.show()
    skewt_volatility = skewt_result.conditional_volatility
    skewt_resid = skewt_result.resid/skewt_volatility
    print(skewt_resid)
    ax = skewt_resid.hist(color='salmon', bins=40)
    ax.set(title='Distribution of Standardized Residuals')
    plt.show()
    s1 = pd.Series(skewt_resid, name='residulas')
    stats.probplot(s1, dist="norm",plot=pylab)
    pylab.show()
    
    ex= get_extremes(s1, "BM", block_size="252D")
    print(ex)
    ###  to find the fit and do the test on models
    model = EVA(s1)
    model.get_extremes(method="BM", block_size="252D")
    model.plot_extremes()
    plt.show()
    model.fit_model()
    model.plot_diagnostic(alpha=0.95)
    plt.show()
    ### if POT is used then thurshload has to be evaluated visually
    p= plot_mean_residual_life(s1)
    p.plot()
    plt.show()
    plot_parameter_stability(s1)
    plt.show()
    plot_return_value_stability(s1, return_period=100, thresholds=np.linspace(1.25, 1.95, 20), alpha=0.95)
    plt.show()
    plot_threshold_stability(s1,return_period=100,thresholds=np.linspace(1.25, 1.95, 20))
    plt.show()
    fitted_gpd = pot.gpd_pot(s1, tu=0.95, fit="mle")
    print(fitted_gpd.Beta, fitted_gpd.Xi)
    q_empirical= fitted_gpd.quantile(q=0.95)
    print(q_empirical)
    pot.mean_exc(s1)
    nu = skewt_result.params[3]

# t distribution skew (lambda)
    lam = skewt_result.params[4]
    

# Forecast from 2020-01-01 onward 
    #start_loc = df[df['Date']=='02-01-2020'].index
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


    # Emperical VaR
    VaR_empirical = mean_forecast.values + np.sqrt(variance_forecast).values * q_empirical
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
    
    return skewt_resid



if __name__ == '__main__':
    res = readData.read("FCX")
    s =evtgarch(res)

