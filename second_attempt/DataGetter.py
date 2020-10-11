#  @author Duke Chain
#  @File:DataGetter.py
#  @createTime 2020/10/09 19:02:09

import pandas as pd
import traceback
import time
import _thread
import tushare as ts
import os
from os.path import join, getsize


def data_daily(pro, stocks):
    # 上证指数
    for stock in stocks:
        target = str(stock) + '.SH'
        print("日线正在尝试", target)
        try:
            data = pro.daily(ts_code=target, start_date='20160809')
            fileName = str(stock) + '_SH'
            filePath = 'D:\\Project_Red\\data_daily\\' + fileName + '.csv'
            data.to_csv(filePath)
            print(target, "日线数据已保存！")
        except Exception as e:
            print(target, "日线尝试失败！")
            print(e.args)
            print(traceback.format_exc())
            continue

    # 深证指数
    for stock in stocks:
        target = str(stock) + '.SZ'
        print("日线正在尝试", target)
        try:
            data = pro.daily(ts_code=target, start_date='20160809')
            fileName = str(stock) + '_SZ'
            filePath = 'D:\\Project_Red\\data_daily\\' + fileName + '.csv'
            data.to_csv(filePath)
            print(target, "日线数据已保存！")
        except Exception as e:
            print(target, "日线尝试失败！")
            print(e.args)
            print(traceback.format_exc())
            continue

    # 文件筛选
    for root, dirs, files in os.walk('D:\\Project_Red\\data_daily'):
        for name in files:
            size = getsize(join(root, name))
            size = size / 1000
            if size < 1:
                os.remove('D:\\Project_Red\\data_daily\\' + name)
    print('日线数据下载完成！')


def data_weekly(pro, stocks):
    # 上证指数
    times = 0
    for stock in stocks:
        target = str(stock) + '.SH'
        print("周线正在尝试", target)
        times += 1
        if times % 200 == 0 and times != 0:
            print('休眠中')
            time.sleep(40)
        try:
            data = pro.weekly(ts_code=target, start_date='20160809')
            fileName = str(stock) + '_SH'
            filePath = 'D:\\Project_Red\\data_weekly\\' + fileName + '.csv'
            data.to_csv(filePath)
            print(target, "周线数据已保存！")
        except Exception as e:
            print(target, "周线尝试失败！")
            print(e.args)
            print(traceback.format_exc())
            continue

    # 深证指数
    for stock in stocks:
        target = str(stock) + '.SZ'
        print("周线正在尝试", target)
        times += 1
        if times % 200 == 0 and times != 0:
            print('休眠中')
            time.sleep(40)
        try:
            data = pro.weekly(ts_code=target, start_date='20160809')
            fileName = str(stock) + '_SZ'
            filePath = 'D:\\Project_Red\\data_weekly\\' + fileName + '.csv'
            data.to_csv(filePath)
            print(target, "周线数据已保存！")
        except Exception as e:
            print(target, "周线尝试失败！")
            print(e.args)
            print(traceback.format_exc())
            continue

    # 文件筛选
    for root, dirs, files in os.walk('D:\\Project_Red\\data_weekly'):
        for name in files:
            size = getsize(join(root, name))
            size = size / 1000
            if size < 1:
                os.remove('D:\\Project_Red\\data_weekly\\' + name)
    print('周线数据下载完成！')


def data_monthly(pro, stocks):
    # 上证指数
    times = 0
    for stock in stocks:
        target = str(stock) + '.SH'
        print("月线正在尝试", target)
        times += 1
        if times % 200 == 0 and times != 0:
            print('休眠中')
            time.sleep(40)
        try:
            data = pro.weekly(ts_code=target, start_date='20160809')
            fileName = str(stock) + '_SH'
            filePath = 'D:\\Project_Red\\data_monthly\\' + fileName + '.csv'
            data.to_csv(filePath)
            print(target, "月线数据已保存！")
        except Exception as e:
            print(target, "月线尝试失败！")
            print(e.args)
            print(traceback.format_exc())
            continue

    # 深证指数
    for stock in stocks:
        target = str(stock) + '.SZ'
        print("月线正在尝试", target)
        times += 1
        if times % 200 == 0 and times != 0:
            print('休眠中')
            time.sleep(40)
        try:
            data = pro.weekly(ts_code=target, start_date='20160809')
            fileName = str(stock) + '_SZ'
            filePath = 'D:\\Project_Red\\data_monthly\\' + fileName + '.csv'
            data.to_csv(filePath)
            print(target, "月线数据已保存！")
        except Exception as e:
            print(target, "月线尝试失败！")
            print(e.args)
            print(traceback.format_exc())
            continue

    # 文件筛选
    for root, dirs, files in os.walk('D:\\Project_Red\\data_monthly'):
        for name in files:
            size = getsize(join(root, name))
            size = size / 1000
            if size < 1:
                os.remove('D:\\Project_Red\\data_monthly\\' + name)
    print('月线数据下载完成！')


if __name__ == '__main__':
    # 接口获取
    ts.set_token('6e32db130ef1ea433f2bf65245ae9026e355cb91d31e47ee50b76598')
    pro = ts.pro_api()
    df = ts.get_stock_basics()
    stocks = df.index.tolist()

    # try:
    #     _thread.start_new_thread(data_daily, (pro, stocks,))
    #     _thread.start_new_thread(data_weekly, (pro, stocks,))
    #     _thread.start_new_thread(data_monthly, (pro, stocks,))
    # except Exception as e:
    #     print(e.args)
    #     print(traceback.format_exc())

    data_daily(pro, stocks)
    time.sleep(60)
    data_weekly(pro, stocks)
    time.sleep(60)
    data_monthly(pro, stocks)
    print("All done")

