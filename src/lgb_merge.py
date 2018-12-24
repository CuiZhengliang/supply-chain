#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: lgb_merge.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/19 16:26
"""
import pandas as pd
import numpy as np

if __name__ == '__main__':
    sample = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\submit_example.csv')
    sample = sample[['sku_id']]
    for i in range(5):
        path = r'E:\PycharmProjects\Task2Plus\lgb_results\weekPlus' + str(i + 1) + '.csv'
        df = pd.read_csv(path)
        title = 'week' + str(i + 1)
        sample = pd.merge(sample,df,on='sku_id')

    sample.to_csv(r'E:\PycharmProjects\Task2Plus\output\lgbfinal.csv',index=False)
