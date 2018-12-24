#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: lgb_week1.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/14 15:56
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

if __name__ == '__main__':
    scaler = MinMaxScaler()
    datas = list()
    target = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\submit_example.csv')
    relation = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goods_sku_relation.csv')
    target = pd.merge(target[['sku_id']], relation, on='sku_id')
    goods_list = target.goods_id.unique()

    for i in range(6):
        trainPath = r'E:\PycharmProjects\Task2Plus\TrainSets\trainAllWithLabel' + str(i + 1) + '.csv'
        traindata = pd.read_csv(trainPath)
        traindata.drop_duplicates(inplace=True)
        datas.append(traindata)
    dataset = pd.concat(datas, axis=0)

    online_test = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TrainSets\trainWithLabel1.csv')
    train_xy, offline_test = train_test_split(dataset, test_size=0.2, random_state=21)
    train, val = train_test_split(train_xy, test_size=0.2, random_state=21)

    y_train = train.week1
    X_train = train[train.columns[2:-5]]
    # X_train = scaler.fit_transform(X_train)

    y_val = val.week1
    X_val = val[val.columns[2:-5]]
    # X_val = scaler.fit_transform(X_val)

    offline_test_X = offline_test[offline_test.columns[2:-5]]
    # offline_test_X = scaler.fit_transform(offline_test_X)
    online_test_X = online_test[online_test.columns[2:-5]]
    # online_test_X = scaler.fit_transform(online_test_X)

    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=False)

    #     params = {
    #         'booting_type': 'gbdt',
    #         'objective': 'regression_l2',
    #         'metric': {'rmse'},
    #     }
    #
    #     print('交叉验证')
    #     min_merror = float('Inf')
    #     best_params = {}
    #
    #     print('调参1：提高准确率')
    #     for num_leaves in range(20, 50, 2):
    #         for max_depth in range(7, 10):
    #             params['num_leaves'] = num_leaves
    #             params['max_depth'] = max_depth
    #
    #             cv_results = lgb.cv(
    #                 params,
    #                 lgb_train,
    #                 seed=2018,
    #                 nfold=3,
    #                 metrics='rmse',
    #                 early_stopping_rounds=20,
    #                 verbose_eval=True
    #             )
    #
    #             mean_merror = pd.Series(cv_results['rmse-mean']).min()
    #             boost_rounds = pd.Series(cv_results['rmse-mean']).argmin()
    #
    #             if mean_merror < min_merror:
    #                 min_merror = mean_merror
    #                 best_params['num_leaves'] = num_leaves
    #                 best_params['max_depth'] = max_depth
    #
    #     params['num_leaves'] = best_params['num_leaves']
    #     params['max_depth'] = best_params['max_depth']
    #
    #     print(params)
    #     result = pd.DataFrame(params, index=range(2))
    #     result.to_csv(r'E:\PycharmProjects\Task2Plus\params\result_params_week1.csv', index=False)
    #
    #     min_merror = float('Inf')
    #     best_params['min_data_in_leaf'] = 0
    #     best_params['max_bin'] = 0
    #
    #     print('调参2：降低过拟合')
    #     best_params['min_data_in_leaf'] = 10
    #     # for max_bin in range(2,5):
    #     for min_data_in_leaf in range(10, 30, 1):
    #         # params['max_bin'] = max_bin
    #         params['min_data_in_leaf'] = min_data_in_leaf
    #
    #         cv_results = lgb.cv(
    #             params,
    #             lgb_train,
    #             seed=42,
    #             nfold=3,
    #             metrics=['rmse'],
    #             early_stopping_rounds=20,
    #             verbose_eval=True
    #         )
    #
    #         mean_merror = pd.Series(cv_results['rmse-mean']).min()
    #         boost_rounds = pd.Series(cv_results['rmse-mean']).argmin()
    #
    #         if mean_merror < min_merror:
    #             min_merror = mean_merror
    #             # best_params['max_bin'] = max_bin
    #             best_params['min_data_in_leaf'] = min_data_in_leaf
    #
    # params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    # # params['max_bin'] = best_params['max_bin']
    # print(params)
    #
    # result = pd.DataFrame(params, index=range(2))
    # result.to_csv(r'E:\PycharmProjects\Task2Plus\params\result_params_week1.csv', index=False)

    params = {
        # 'task': 'train',
        'boosting': 'gbdt',  # 设置提升类型
        'application': 'regression_l2',  # 目标函数
        'metric': 'rmse',  # 评估函数
        'max_depth': 9,
        # 'min_data':30,
        'num_leaves': 46,  # 叶子节点数
        'learning_rate': 0.03,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        # 'max_bin': 10,
        'min_data_in_leaf': 19,
        # 'bagging_fraction': 0.8, # 建树的样本采样比例
        # 'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    #
    params['learning_rate'] = 0.01
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=5000,
        early_stopping_rounds=100,
    )
    #
    # print('线下预测')
    # preds_offline = gbm.predict(X_val,num_iteration=gbm.best_iteration)
    # val['pre'] = pd.Series(preds_offline)
    #
    # preds_online = gbm.predict(online_test_X, num_iteration=gbm.best_iteration)
    # online_test['pre'] = pd.Series(preds_online)
    #
    # # print('log_loss',metrics.mean_squared_error(offline_test['week1'],offline_test['pre'])** 0.5)
    #
    # # print('线上预测')
    # # pres_online = gbm.predict(online_test_X,num_iteration=gbm.best_iteration)
    # # online_test['pre'] = pd.Series(pres_online)
    #
    # df = pd.DataFrame(dataset.columns[2:-5].tolist(), columns=['feature'])
    # df['importance'] = pd.Series((gbm.feature_importance()))
    # df = df.sort_values(by='importance', ascending=False)
    # df.to_csv("feature_score.csv", index=None)
    #
    # val[['sku_id','week1','pre']].to_csv(r'E:\PycharmProjects\Task2Plus\var_test\val_week1.csv',index=False)
    # online_test[['sku_id','week1','pre']].to_csv(r'E:\PycharmProjects\Task2Plus\var_test\online_week1.csv',index=False)

    test = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\TestAllPlus.csv')
    test.fillna(0,inplace=True)
    test[test.columns[-11:]] = test[test.columns[-11:]].astype('int')
    # X_test = scaler.fit_transform(test[test.columns[2:]])
    preds_test = gbm.predict(test[test.columns[2:]])
    test['week1'] = pd.Series(preds_test)
    test[['sku_id', 'week1']].to_csv(r'E:\PycharmProjects\Task2Plus\lgb_results\weekPlus1.csv', index=False)
