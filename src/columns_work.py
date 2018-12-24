#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: columns_work.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/21 18:57
"""
import pandas as pd
if __name__ == '__main__':
    for i in range(1):
        sale_feature_file_name = r'E:\PycharmProjects\Task2Plus\TestSets\TestDaily7feature.csv'
        sale7feature = pd.read_csv(sale_feature_file_name)
        column = sale7feature.columns.tolist()
        for j in range(1, len(column)):
            sale7feature.rename(columns={column[j]: column[j] + str(7)}, inplace=True)
        sale7feature.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\dailytest7days.csv', index=False)
        print('Feature' + str(i + 1) + 'finished!')

    for i in range(1):
        sale_feature_file_name = r'E:\PycharmProjects\Task2Plus\TestSets\TestSale7feature.csv'
        sale7feature = pd.read_csv(sale_feature_file_name)
        column = sale7feature.columns.tolist()
        for j in range(2, len(column)):
            sale7feature.rename(columns={column[j]: column[j] + str(7)}, inplace=True)
        sale7feature.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\saletest7days.csv', index=False)
        print('Feature' + str(i + 1) + 'finished!')
