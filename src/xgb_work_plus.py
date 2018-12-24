#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: xgb_work_plus.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/21 19:40
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

params={'booster':'gbtree',
	    'objective': 'reg:linear',
	    'eval_metric':'rmse',
	    'gamma':0.05,
	    'min_child_weight':0.8,
	    'max_depth':8,
	    # 'lambda':10,
	    'subsample':0.8,
	    'colsample_bytree':0.6,
	    'colsample_bylevel':0.8,
	    'eta': 0.2,
	    'tree_method':'exact',
	    'seed':0,
	    # 'nthread':12
}

if __name__ == '__main__':
    datas = list()
    for i in range(6):
        trainPath = r'E:\PycharmProjects\Task2Plus\TrainSets\trainAllWithLabel' + str(i + 1) + '.csv'
        train = pd.read_csv(trainPath)
        train.fillna(0,inplace=True)
        datas.append(train)
    dataset = pd.concat(datas, axis=0)
    test = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\TestAllPlus.csv')
    test.fillna(0,inplace=True)
    # target = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\submit_example.csv')
    # test = pd.merge(test,target,on='sku_id')

    target = test[['sku_id']]

    #for week1
    X = dataset[dataset.columns[2:-5]]
    Y = dataset['week1']

    DataSet = xgb.DMatrix(X,label=Y)
    dataset3 = xgb.DMatrix(test[test.columns[2:]])

    watchlist = [(DataSet,'train')]
    model = xgb.train(params,DataSet,num_boost_round=500,evals=watchlist)
    y_pred = model.predict(dataset3)
    target['week1'] = pd.Series(y_pred)
    print('finished')
    target.to_csv('xgb.csv',index=False)


    #for week2
    X = dataset[dataset.columns[2:-5]]
    Y = dataset['week2']

    DataSet = xgb.DMatrix(X,label=Y)
    dataset3 = xgb.DMatrix(test[test.columns[2:]])

    watchlist = [(DataSet,'train')]
    model = xgb.train(params,DataSet,num_boost_round=500,evals=watchlist)
    y_pred = model.predict(dataset3)
    target['week2'] = pd.Series(y_pred)

    #for week3
    X = dataset[dataset.columns[2:-5]]
    Y = dataset['week3']

    DataSet = xgb.DMatrix(X,label=Y)
    dataset3 = xgb.DMatrix(test[test.columns[2:]])

    watchlist = [(DataSet,'train')]
    model = xgb.train(params,DataSet,num_boost_round=500,evals=watchlist)
    y_pred = model.predict(dataset3)
    target['week3'] = pd.Series(y_pred)

    #for week4
    X = dataset[dataset.columns[2:-5]]
    Y = dataset['week4']

    DataSet = xgb.DMatrix(X,label=Y)
    dataset3 = xgb.DMatrix(test[test.columns[2:]])

    watchlist = [(DataSet,'train')]
    model = xgb.train(params,DataSet,num_boost_round=500,evals=watchlist)
    y_pred = model.predict(dataset3)
    target['week4'] = pd.Series(y_pred)

    #for week5
    X = dataset[dataset.columns[2:-5]]
    Y = dataset['week5']

    DataSet = xgb.DMatrix(X,label=Y)
    dataset3 = xgb.DMatrix(test[test.columns[2:]])

    watchlist = [(DataSet,'train')]
    model = xgb.train(params,DataSet,num_boost_round=500,evals=watchlist)
    y_pred = model.predict(dataset3)
    target['week5'] = pd.Series(y_pred)

    target.to_csv(r'E:\PycharmProjects\Task2Plus\output\xgbAll.csv',index=False)
