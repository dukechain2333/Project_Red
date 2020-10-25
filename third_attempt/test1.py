#  @author Duke Chain
#  @File:test1.py
#  @createTime 2020/10/16 16:54:16

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

name='002521_SZ.csv'
df = pd.read_csv('D:\\Project_Red\\data_monthly\\' + name)
s = pd.Series(df['amount'])
pd.plotting.lag_plot(s, lag=1)
picName = name.split('.')[0]
plt.title("lag_monthly: " + picName)
plt.savefig('D:\\Project_Red\\lag_picture\\lag_monthly\\' + picName + '.png')

