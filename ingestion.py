import pandas as pd
import sys

data = pd.read_csv('portfolio.csv')
print(data.fcx_return[0:10])