#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: promote_process.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/10 15:06
"""
import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    promote = pd.read_csv(r'E:\PycharmProjects\Task2\dataset\promotePlus.csv')
    start_date = datetime.strptime(str(20170510),'%Y%m%d')
    end_date = datetime.strptime(str(20180317),'%Y%m%d')

    promote.promote_start_time = pd.to_datetime(promote.promote_start_time,format='%Y-%m-%d %H:%M:%S')
    promote.promote_end_time = pd.to_datetime(promote.promote_end_time,format='%Y-%m-%d %H:%M:%S')
    promote.start = pd.to_datetime(promote.start,format='%Y-%m-%d')
    promote.end = pd.to_datetime(promote.end, format='%Y-%m-%d')

    promote = promote[(promote.promote_start_time >= start_date) & (promote.promote_end_time <= end_date)]

    promote = promote[promote.columns[1:]].drop_duplicates()

    promote.to_csv(r'E:\PycharmProjects\Task2\dataset\target_promote.csv',index=False)
