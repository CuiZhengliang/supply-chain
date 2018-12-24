#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: lasso_work.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/15 16:57
"""
import pandas as pd
import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
scaler = MinMaxScaler()
datas = list()
target = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\submit_example.csv')
relation = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goods_sku_relation.csv')
target = pd.merge(target[['sku_id']], relation, on='sku_id')
goods_list = target.goods_id.unique()

for i in range(1, 6):
    trainPath = r'E:\PycharmProjects\Task2Plus\TrainSets\trainWithLabel' + str(i + 1) + '.csv'
    traindata = pd.read_csv(trainPath)
    traindata.drop_duplicates(inplace=True)
    datas.append(traindata)
dataset = pd.concat(datas, axis=0)

online_test = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TrainSets\trainWithLabel1.csv')
train_xy, offline_test = train_test_split(dataset, test_size=0.2, random_state=21)
train, val = train_test_split(train_xy, test_size=0.2, random_state=21)

y_train = train.week1
X_train = train[train.columns[2:-5]]
X_train = scaler.fit_transform(X_train)

y_val = val.week1
X_val = val[val.columns[2:-5]]
X_val = scaler.fit_transform(X_val)

model = Lasso(alpha=0.01)
model.fit(X_train,y_train)

predicted = model.predict(X_val)
val['pre'] = pd.Series(predicted)
