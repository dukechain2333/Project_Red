#  @author Duke Chain
#  @File:LagGenerate.py
#  @createTime 2020/10/10 20:25:10


import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import lag_plot
import os


def lag_daily():
    for root, dirs, files in os.walk('D:\\Project_Red\\data_daily'):
        for name in files:
            print("正在绘制lag_daliy", name)
            df = pd.read_csv('D:\\Project_Red\\data_daily\\' + name)
            s = pd.Series(df['amount'])
            pd.plotting.lag_plot(s, lag=1)
            picName = name.split('.')[0]
            plt.title("lag_daily: " + picName)
            plt.savefig('D:\\Project_Red\\lag_picture\\lag_daily\\' + picName + '.png')


def lag_weekly():
    for root, dirs, files in os.walk('D:\\Project_Red\\data_weekly'):
        for name in files:
            print("正在绘制lag_weekly", name)
            df = pd.read_csv('D:\\Project_Red\\data_weekly\\' + name)
            s = pd.Series(df['amount'])
            pd.plotting.lag_plot(s, lag=1)
            picName = name.split('.')[0]
            plt.title("lag_weekly: " + picName)
            plt.savefig('D:\\Project_Red\\lag_picture\\lag_weelky\\' + picName + '.png')


def lag_monthly():
    for root, dirs, files in os.walk('D:\\Project_Red\\data_monthly'):
        for name in files:
            print("正在绘制lag_monthly", name)
            df = pd.read_csv('D:\\Project_Red\\data_monthly\\' + name)
            s = pd.Series(df['amount'])
            pd.plotting.lag_plot(s, lag=1)
            picName = name.split('.')[0]
            plt.title("lag_monthly: " + picName)
            plt.savefig('D:\\Project_Red\\lag_picture\\lag_monthly\\' + picName + '.png')


if __name__ == '__main__':
    lag_daily()
    lag_weekly()
    lag_monthly()
    print("all done!")
