#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: data_merge.py
@e-mail: cauchyguo@gmail.com
@time: 2018/10/4 8:41
"""
import pandas as pd


if __name__ == '__main__':
    # 修正data
    # for i in range(8):
    #     featurepath = r'E:\PycharmProjects\Task2Plus\features\dailyfeature' + str(i+1) + '.csv'
    #     feature = pd.read_csv(featurepath)
    #     feature.rename(columns={'goods_total_click_x':'goods_total_click','goods_total_click_y':'goods_cart_total_click'},inplace=True)
    #     feature.to_csv(featurepath,index=False)

    # # 合并特征
    # for i in range(6):
    #     dailyfeaturepath = r'E:\PycharmProjects\Task2Plus\features\daily7feature' + str(i + 1) + '.csv'
    #     salefeaturepath = r'E:\PycharmProjects\Task2Plus\features\sale7feature' + str(i + 1) + '.csv'
    #     d = pd.read_csv(dailyfeaturepath)
    #     s = pd.read_csv(salefeaturepath)
    #     df = pd.merge(s, d, on='goods_id')
    #     trainPath = r'E:\PycharmProjects\Task2Plus\TrainSets\train7' + str(i + 1) + '.csv'
    #     df.to_csv(trainPath,index=False)

    # Testdaily  = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\dailytest7days.csv')
    # Testsale = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\saletest7days.csv')
    # df = pd.merge(Testsale, Testdaily, on='goods_id')
    # df.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\saledailyTest7.csv', index=False)
    #

    # Strategy = {
    #     'goods_promote_min_discount_rate': 1,
    #     'goods_promote_max_discount_rate': 1,
    #     'goods_promote_avg_discount_rate': 1,
    #     'goods_promote_total_dates': 0,
    #     'goods_promote_max_period': 0,
    #     'goods_promote_min_period': 0,
    #     'goods_promote_weekends_num': 0,
    #     'goods_promote_holidays_num': 0,
    #     'goods_promote_holidays_fisrtday': -1,
    #     'goods_promote_holidays_lastday': -1,
    # }
    #
    # for i in range(6):
    #     propath = r'E:\PycharmProjects\Task2Plus\features\promotefeature' + str(i + 1) + '.csv'
    #     trapath = r'E:\PycharmProjects\Task2Plus\TrainSets\train' + str(i + 1) + '.csv'
    #
    #     tosale = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\totalsale.csv')
    #     promote = pd.read_csv(propath)
    #     train = pd.read_csv(trapath)
    #
    #     df = pd.merge(train,promote,on='goods_id',how='left')
    #     df['goods_is_promoted'] = 0
    #     df['goods_is_promoted'][df.goods_promote_min_discount_rate.notna()] = 1
    #
    #     df.fillna(Strategy,inplace=True)
    #
    #     df.goods_promote_holidays_fisrtday = df.goods_promote_holidays_fisrtday.add(1)
    #     df.goods_promote_holidays_lastday = df.goods_promote_holidays_lastday.add(1)
    #
    #     # firstday_dummies = pd.get_dummies(df.goods_promote_holidays_fisrtday)
    #     # lastday_dummies = pd.get_dummies(df.goods_promote_holidays_fisrtday)
    #     #
    #     # firstday_dummies.columns = ['holidays_fisrtday_weekday' + str(i + 1) for i in range(firstday_dummies.shape[1])]
    #     # lastday_dummies.columns = ['holidays_lastday_weekday' + str(i + 1) for i in range(lastday_dummies.shape[1])]
    #     # df = pd.concat([df, firstday_dummies,lastday_dummies], axis=1)
    #     #
    #     # del df['goods_promote_holidays_fisrtday']
    #     # del df['goods_promote_holidays_lastday']
    #
    #     df = pd.merge(df,tosale,on='sku_id')
    #     writepath = r'E:\PycharmProjects\Task2Plus\TrainSets\trainPlus' + str(i + 1) + '.csv'
    #     df.to_csv(writepath,index=False)
    #
    # TestPromote = pd.read_csv(r'E:\PycharmProjects\Task2Plus\features\TestPromotefeature.csv')
    # test = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\Test.csv')
    # tosale = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\totalsale.csv')
    #
    # df = pd.merge(test, TestPromote, on='goods_id', how='left')
    #
    # df['goods_is_promoted'] = 0
    # df['goods_is_promoted'][df.goods_promote_min_discount_rate.notna()] = 1
    #
    # df.fillna(Strategy, inplace=True)
    #
    # df.goods_promote_holidays_fisrtday = df.goods_promote_holidays_fisrtday.add(1)
    # df.goods_promote_holidays_lastday = df.goods_promote_holidays_lastday.add(1)
    # #
    # # firstday_dummies = pd.get_dummies(df.goods_promote_holidays_fisrtday)
    # # lastday_dummies = pd.get_dummies(df.goods_promote_holidays_fisrtday)
    # #
    # # firstday_dummies.columns = ['holidays_fisrtday_weekday' + str(i + 1) for i in range(firstday_dummies.shape[1])]
    # # lastday_dummies.columns = ['holidays_lastday_weekday' + str(i + 1) for i in range(lastday_dummies.shape[1])]
    # # df = pd.concat([df, firstday_dummies, lastday_dummies], axis=1)
    # #
    # # del df['goods_promote_holidays_fisrtday']
    # # del df['goods_promote_holidays_lastday']
    #
    # df = pd.merge(df, tosale, on='sku_id')
    # df.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\TestPlus.csv', index=False)


    # 合并标签

    # for i in range(6):
    #     trainPath = r'E:\PycharmProjects\Task2Plus\TrainSets\trainWithLabel' + str(i + 1) + '.csv'
    #     train7p = r'E:\PycharmProjects\Task2Plus\TrainSets\train7' + str(i + 1) + '.csv'
    #     train = pd.read_csv(trainPath)
    #     train7 = pd.read_csv(train7p)
    #     finaltrain = pd.merge(train[train.columns[:-5]],train7,on=['sku_id','goods_id'],how='left')
    #     fileName = r'E:\PycharmProjects\Task2Plus\TrainSets\Trainpplus' + str(i + 1) + '.csv'
    #     finaltrain.to_csv(fileName,index=False)


    Test7 = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\saledailyTest7.csv')
    Test = pd.read_csv(r'E:\PycharmProjects\Task2Plus\TestSets\TestPlus.csv')

    test = pd.merge(Test, Test7, on=['sku_id','goods_id'],how='left')
    fileName = r'E:\PycharmProjects\Task2Plus\TestSets\TestAllPlus.csv'
    test.to_csv(fileName,index=False)


    #
    # for i in range(6):
    #     trainPath = r'E:\PycharmProjects\Task2Plus\TrainSets\Trainpplus' + str(i + 1) + '.csv'
    #     labelPath = r'E:\PycharmProjects\Task2Plus\LabelSets\label' + str(i + 1) + '.csv'
    #     train = pd.read_csv(trainPath)
    #     label = pd.read_csv(labelPath)
    #     finaltrain = pd.merge(train,label,on='sku_id')
    #     fileName = r'E:\PycharmProjects\Task2Plus\TrainSets\trainAllWithLabel' + str(i + 1) + '.csv'
    #     finaltrain.to_csv(fileName,index=False)

