#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: lgb_work.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/8 13:12
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

params = {
    # 'task': 'train',
    'boosting': 'gbdt',  # 设置提升类型
    'application': 'regression_l2', # 目标函数
    'metric': {'rmse'},  # 评估函数
    'max_depth':8,
    'min_data':30,
    # 'num_leaves': 150,   # 叶子节点数
    'learning_rate': 0.03,  # 学习速率
    'feature_fraction': 0.8, # 建树的特征选择比例
    # 'bagging_fraction': 0.8, # 建树的样本采样比例
    # 'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    # 'device':'gpu',
}
if __name__ == '__main__':
    datas = list()
    target = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\submit_example.csv')
    relation = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goods_sku_relation.csv')
    target = pd.merge(target[['sku_id']],relation,on='sku_id')
    goods_list = target.goods_id.unique()
    scaler = MinMaxScaler()

    for i in range(6):
        trainPath = r'E:\PycharmProjects\Task2Plus\TrainSets\trainWithLabel' + str(i + 1) + '.csv'
        traindata = pd.read_csv(trainPath)
        datas.append(traindata[traindata.goods_id.isin(goods_list)])

    dataset = pd.concat(datas, axis=0)
    test = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\TestPlus.csv')
    # test = pd.merge(test,target,on='sku_id')


    # for week1
    X = dataset[dataset.columns[2:-5]]
    X = scaler.fit_transform(X)
    Y = dataset['week1']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train,label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100)
    gbm.save_model(r'E:\PycharmProjects\Task2Plus\models\model1.txt')
    y_pred = gbm.predict(test[test.columns[2:]],num_iteration=gbm.best_iteration)
    # y_pred = y_pred.astype('int')
    test['week1'] = pd.Series(y_pred)

    # for week2
    X = dataset[dataset.columns[2:-5]]
    X = scaler.fit_transform(X)
    Y = dataset['week2']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100)
    gbm.save_model(r'E:\PycharmProjects\Task2Plus\models\model2.txt')
    y_pred = gbm.predict(test[test.columns[2:]],num_iteration=gbm.best_iteration)
    # y_pred = y_pred.astype('int')
    test['week2'] = pd.Series(y_pred)

    # for week3
    X = dataset[dataset.columns[2:-5]]
    X = scaler.fit_transform(X)
    Y = dataset['week3']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100)
    gbm.save_model(r'E:\PycharmProjects\Task2Plus\models\model3.txt')
    y_pred = gbm.predict(test[test.columns[2:]],num_iteration=gbm.best_iteration)
    # y_pred = y_pred.astype('int')
    test['week3'] = pd.Series(y_pred)

    # for week4
    X = dataset[dataset.columns[2:-5]]
    X = scaler.fit_transform(X)
    Y = dataset['week4']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100)
    gbm.save_model(r'E:\PycharmProjects\Task2Plus\models\model4.txt')
    y_pred = gbm.predict(test[test.columns[2:]],num_iteration=gbm.best_iteration)
    # y_pred = y_pred.astype('int')
    test['week4'] = pd.Series(y_pred)

    # for week5
    X = dataset[dataset.columns[2:-5]]
    X = scaler.fit_transform(X)
    Y = dataset['week5']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100)
    gbm.save_model(r'E:\PycharmProjects\Task2Plus\models\model5.txt')
    y_pred = gbm.predict(test[test.columns[2:]],num_iteration=gbm.best_iteration)
    # y_pred = y_pred.astype('int')
    test['week5'] = pd.Series(y_pred)

    test = test[['sku_id','week1','week2','week3','week4','week5']]
    test.to_csv(r'E:\PycharmProjects\Task2Plus\output\10193.csv',index=False)