#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: label_biu.py
@e-mail: cauchyguo@gmail.com
@time: 2018/9/28 14:22
"""
import pandas as pd
from datetime import datetime


def date_split(date, startTime):
    period = (date - startTime).days
    week = period // 7 + 1
    if week >= 5:
        return 5
    else:
        return week


if __name__ == '__main__':
    for i in range(6):
        LabelDataPath = r'E:\PycharmProjects\Task2Plus\LabelSets\labelset' + str(i + 1) + '.csv'
        df = pd.read_csv(LabelDataPath)
        start_date = datetime.strptime(str(min(df.data_date)), '%Y%m%d')
        df['date'] = pd.to_datetime(df.data_date, format='%Y%m%d')
        df['targetWeek'] = df['date'].apply(date_split, startTime=start_date)
        df = df[['sku_id', 'goods_num', 'targetWeek']].groupby(['sku_id', 'targetWeek']).agg('sum').reset_index()
        for j in range(5):
            title = 'week' + str(j + 1)
            df[title] = 0
        for j in range(5):
            title = 'week' + str(j + 1)
            df[title] = df.apply(lambda x: x['goods_num'] if x['targetWeek'] == j + 1 else 0, axis=1)
        df = df[['sku_id', 'week1', 'week2', 'week3', 'week4', 'week5']]
        df = df.groupby('sku_id').agg('sum').reset_index()
        writepath = r'E:\PycharmProjects\Task2Plus\LabelSets\label' + str(i + 1) + '.csv'
        df.to_csv(writepath, index=False)

    # df = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goodsalePlus.csv')
    # df = df[(df.data_date >= 20170522) & (df.data_date <= 20170625)]
    # start_date = datetime.strptime(str(min(df.data_date)), '%Y%m%d')
    # df['date'] = pd.to_datetime(df.data_date, format='%Y%m%d')
    # df['targetWeek'] = df['date'].apply(date_split, startTime=start_date)
    # df = df[['sku_id', 'goods_num', 'targetWeek']].groupby(['sku_id', 'targetWeek']).agg('sum').reset_index()
    # for j in range(5):
    #     title = 'week' + str(j + 1)
    #     df[title] = 0
    # for j in range(5):
    #     title = 'week' + str(j + 1)
    #     df[title] = df.apply(lambda x: x['goods_num'] if x['targetWeek'] == j + 1 else 0, axis=1)
    # df = df[['sku_id', 'week1', 'week2', 'week3', 'week4', 'week5']]
    # df = df.groupby('sku_id').agg('sum').reset_index()
    # target = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\submit_example.csv')
    # df = pd.merge(target[['sku_id']],df,on='sku_id')
    # df.to_csv(r'E:\PycharmProjects\Task2Plus\output\Last.csv', index=False)
