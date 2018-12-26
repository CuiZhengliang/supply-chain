#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: run_all.py
@e-mail: cauchyguo@gmail.com
@time: 2018/12/26 0:27
"""
import os
import datetime
import time

start_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
print('start time :', start_time)

print("数据预处理中……")
os.system("python src/sedwork.py")
print("划窗中……")
os.system("python src/data_split.py")
print("打标中……")
os.system("python src/label_biu.py")
print("特征提取中……")
os.system("python src/feature_extract.py")
print("lgb模型处理中……")
os.system("python src/lgb_work.py")

end_time = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '%Y-%m-%d %H:%M:%S')
print('end time :', end_time)

run_time = end_time - start_time
print('run time :', run_time)
