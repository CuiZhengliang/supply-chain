#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: data_view.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/9 20:30
"""
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # sale = pd.read_csv(r'E:\PycharmProjects\supply-chain\dataset\goodsalePlus.csv')
    # plt.title('sale information')
    market = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\marketing.csv')
    # sale = sale[['data_date','goods_num']].groupby('data_date').agg('sum').reset_index()
    market.data_date = pd.to_datetime(market.data_date,format='%Y%m%d')
    plt.title('Marketing Plan')
    plt.plot(market.data_date,market['plan'])
    # plt.plot(market.data_date,market['marketing'])
    # sale.data_date = pd.to_datetime(sale.data_date,format='%Y%m%d')
    # plt.plot(pd.to_datetime(sale.data_date.iloc[244:,], format='%Y%m%d'), sale.goods_num.iloc[244:,])


    # daily = pd.read_csv(r'E:\PycharmProjects\supply-chain\dataset\goodsdaily.csv')
    # df = daily.drop(['goods_id'],axis=1)
    # df = df.groupby('data_date').agg('sum').reset_index()
    # df.data_date = pd.to_datetime(df.data_date,format='%Y%m%d')
    #
    #
    # dates = [20170101,20170424,20170525,20170625,20170626,20170627,20170831,
    #          20170901,20170902,20170903,20170921,20171130,20171201,20171202,
    #          20171203,20180101,20180414,20180501,20180517,20180602]
    #
    # dateList = pd.to_datetime(pd.Series(dates), format='%Y%m%d')
    # dateList = dateList.map(lambda x:x.date())
    #
    # Y = df[df.data_date.isin(dateList)]
    #
    # Y.data_date = pd.to_datetime(Y.data_date,format='%Y%m%d')

    # x = pd.DataFrame()
    # x['data_date'] = df.apply(
    #     lambda x: x['data_date'] if x['data_date'].dayofweek == 4 or x['data_date'].dayofweek == 5 else 8, axis=1)
    # x['sales_uv'] = df.apply(
    #     lambda x: x['sales_uv'] if x['data_date'].dayofweek == 4 or x['data_date'].dayofweek == 5 else 0, axis=1)
    # X = x[x.data_date != 8]
    print('stop here!')
    # plt.plot(df.data_date, df.sales_uv,X.data_date,X.sales_uv,'r+')
    # plt.plot(df.data_date, df.sales_uv, Y.data_date, Y.sales_uv, 'r+')
    plt.show()
