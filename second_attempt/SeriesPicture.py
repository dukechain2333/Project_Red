#  @author Duke Chain
#  @File:SeriesPicture.py
#  @createTime 2020/10/10 21:55:10

import matplotlib.pyplot as plt
import pandas as pd
import os


def daily_series():
    for root, dirs, files in os.walk('D:\\Project_Red\\data_daily'):
        for name in files:
            plt.clf()
            print("正在绘制series_daily", name)
            df = pd.read_csv('D:\\Project_Red\\data_daily\\' + name)
            s = pd.Series(df['amount'])
            picName = name.split('.')[0]
            s.plot()
            plt.title("series_daily:" + picName)
            plt.savefig('D:\\Project_Red\\series_picture\\series_daily\\' + picName + '.png')


def weekly_series():
    for root, dirs, files in os.walk('D:\\Project_Red\\data_weekly'):
        for name in files:
            plt.clf()
            print("正在绘制series_weekly", name)
            df = pd.read_csv('D:\\Project_Red\\data_weekly\\' + name)
            s = pd.Series(df['amount'])
            picName = name.split('.')[0]
            s.plot()
            plt.title("series_weekly:" + picName)
            plt.savefig('D:\\Project_Red\\series_picture\\series_weekly\\' + picName + '.png')


def monthly_series():
    for root, dirs, files in os.walk('D:\\Project_Red\\data_monthly'):
        for name in files:
            plt.clf()
            print("正在绘制series_monthly", name)
            df = pd.read_csv('D:\\Project_Red\\data_monthly\\' + name)
            s = pd.Series(df['amount'])
            picName = name.split('.')[0]
            s.plot()
            plt.title("series_monthly:" + picName)
            plt.savefig('D:\\Project_Red\\series_picture\\series_monthly\\' + picName + '.png')


if __name__ == '__main__':
    daily_series()
    weekly_series()
    monthly_series()
    print('all done!')
