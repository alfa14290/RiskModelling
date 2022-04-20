import pandas as pd
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv('portfolio.csv')
## reverse the frame and change the data
##ToDo: When we extend the portfolio and take some european stocks we have to clean the data for e.x account for trading days
res = data[::-1].reset_index(drop=True)
res['return_fcx'] = res['fcx_price'].pct_change().fillna(method='bfill')
res['return_nktr'] = res['nktr_price'].pct_change().fillna(method='bfill')
print(res.tail())
print(res['return_fcx'].describe())
fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
### Action: Remove to comments to see the normal plot instead of prob plot
#res['return_fcx'].hist(bins=50, ax=ax1)
ax1.set_xlabel('Return')
ax1.set_ylabel('Sample')
ax1.set_title('Return distribution')
#plt.show()
stats.probplot(res['return_fcx'], dist='norm', plot=ax1)
plt.show()