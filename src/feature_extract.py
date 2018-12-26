#! /usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: Cauchy
@file: feature_extract.py
@e-mail: cauchyguo@gmail.com
@time: 2018/9/26 19:54
"""
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from datetime import datetime


def var_value_cal(List, nums=7):
    '''计算方差'''
    avg = sum(List[List.columns[-1]]) / nums
    result = 0
    for i in range(len(List)):
        result += (List[List.columns[-1]].iloc[i] - avg) ** 2
    result += (nums - len(List)) * avg ** 2
    return result / nums


# feature about goods

# 1 for goods click

def goods_total_click_cal(df=pd.DataFrame()):
    """计算Goods7天内总共的点击数"""
    frame = df[['goods_id', 'goods_click']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'goods_click': 'goods_total_click'}, inplace=True)
    return frame


def goods_max_click_cal(df=pd.DataFrame()):
    """计算Goods7天内的最大点击数"""
    frame = df[['goods_id', 'goods_click']].groupby('goods_id').agg('max').reset_index()
    frame.rename(columns={'goods_click': 'goods_max_click'}, inplace=True)
    return frame


def goods_min_click_cal(df=pd.DataFrame()):
    """计算Goods7天内的最小点击数"""
    frame = df[['goods_id', 'goods_click', 'data_date']].groupby('goods_id').agg(
        {'goods_click': 'min', 'data_date': 'count'}).reset_index()
    frame.rename(columns={'goods_click': 'goods_min_click'}, inplace=True)
    frame['goods_min_click'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['goods_min_click'], axis=1)
    return frame[['goods_id', 'goods_min_click']]


def goods_avg_click_cal(df=pd.DataFrame()):
    """计算Goods7天内的平均点击数"""
    frame = df[['goods_id', 'goods_click']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'goods_click': 'goods_avg_click'}, inplace=True)
    frame['goods_avg_click'] = frame['goods_avg_click'] / 7
    return frame


def goods_var_click_cal(df=pd.DataFrame()):
    """计算Goods7天内的点击数方差"""
    frame = df[['goods_id', 'goods_click']].groupby('goods_id').apply(var_value_cal).reset_index()
    frame.columns = ['goods_id', 'goods_var_click']
    return frame


goods_nomal_click_relat = [goods_total_click_cal, goods_max_click_cal,
                           goods_min_click_cal, goods_avg_click_cal, goods_var_click_cal]


# ----------------------------------------------------------------------------------------------------------#

# 2 for goods cart click

def goods_cart_total_click_cal(df=pd.DataFrame()):
    """计算goods 7天内总共的添加购物车数"""
    frame = df[['goods_id', 'cart_click']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'cart_click': 'goods_cart_total_click'}, inplace=True)
    return frame


def goods_cart_max_click_cal(df=pd.DataFrame()):
    """计算goods 7天内最大的添加购物车数"""
    frame = df[['goods_id', 'cart_click']].groupby('goods_id').agg('max').reset_index()
    frame.rename(columns={'cart_click': 'goods_cart_max_click'}, inplace=True)
    return frame


def goods_cart_min_click_cal(df=pd.DataFrame()):
    """计算goods 7天内最小的添加购物车数"""
    frame = df[['goods_id', 'cart_click', 'data_date']].groupby('goods_id').agg(
        {'cart_click': 'min', 'data_date': 'count'}).reset_index()
    frame.rename(columns={'cart_click': 'goods_cart_min_click'}, inplace=True)
    frame['goods_cart_min_click'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['goods_cart_min_click'],
                                                axis=1)
    return frame[['goods_id', 'goods_cart_min_click']]


def goods_cart_avg_click_cal(df=pd.DataFrame()):
    """计算goods 7天内平均的添加购物车数"""
    frame = df[['goods_id', 'cart_click']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'cart_click': 'goods_cart_avg_click'}, inplace=True)
    frame['goods_cart_avg_click'] = frame['goods_cart_avg_click'] / 7
    return frame


def goods_cart_var_click_cal(df=pd.DataFrame()):
    """计算goods 7天内每天添加购物车数的方差"""
    frame = df[['goods_id', 'cart_click']].groupby('goods_id').apply(var_value_cal).reset_index()
    frame.columns = ['goods_id', 'goods_cart_var_click']
    return frame


goods_cart_click_relat = [goods_cart_total_click_cal, goods_cart_max_click_cal,
                          goods_cart_min_click_cal, goods_cart_avg_click_cal, goods_cart_var_click_cal]


# ----------------------------------------------------------------------------------------------------------#

# 3 for goods favorites click

def goods_favor_total_click_cal(df=pd.DataFrame()):
    """计算goods 7天内总总收藏次数"""
    frame = df[['goods_id', 'favorites_click']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'favorites_click': 'goods_favor_total_click'}, inplace=True)
    return frame


def goods_favor_max_click_cal(df=pd.DataFrame()):
    """计算goods 7天内最大的添加购物车数"""
    frame = df[['goods_id', 'favorites_click']].groupby('goods_id').agg('max').reset_index()
    frame.rename(columns={'favorites_click': 'goods_favor_max_click'}, inplace=True)
    return frame


def goods_favor_min_click_cal(df=pd.DataFrame()):
    """计算goods 7天内最小的添加购物车数"""
    frame = df[['goods_id', 'favorites_click', 'data_date']].groupby('goods_id').agg(
        {'favorites_click': 'min', 'data_date': 'count'}).reset_index()
    frame.rename(columns={'favorites_click': 'goods_favor_min_click'}, inplace=True)
    frame['goods_favor_min_click'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['goods_favor_min_click'],
                                                 axis=1)
    return frame[['goods_id', 'goods_favor_min_click']]


def goods_favor_avg_click_cal(df=pd.DataFrame()):
    """计算goods 7天内平均的添加购物车数"""
    frame = df[['goods_id', 'favorites_click']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'favorites_click': 'goods_favor_avg_click'}, inplace=True)
    frame['goods_favor_avg_click'] = frame['goods_favor_avg_click'] / 7
    return frame


def goods_favor_var_click_cal(df=pd.DataFrame()):
    """计算goods 7天内每天添加购物车数的方差"""
    frame = df[['goods_id', 'favorites_click']].groupby('goods_id').apply(var_value_cal).reset_index()
    frame.columns = ['goods_id', 'goods_favor_var_click']
    return frame


goods_favor_click_relat = [goods_favor_total_click_cal, goods_favor_max_click_cal,
                           goods_favor_min_click_cal, goods_favor_avg_click_cal, goods_favor_var_click_cal]


# ----------------------------------------------------------------------------------------------------------#

# 4 for sales uv
def goods_sales_total_uv_cal(df=pd.DataFrame()):
    """计算goods 7天内总购买人数"""
    frame = df[['goods_id', 'sales_uv']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'sales_uv': 'goods_sales_total_uv'}, inplace=True)
    return frame


def goods_sales_max_uv_cal(df=pd.DataFrame()):
    """计算goods 7天内最大的购买人数"""
    frame = df[['goods_id', 'sales_uv']].groupby('goods_id').agg('max').reset_index()
    frame.rename(columns={'sales_uv': 'goods_sales_max_uv'}, inplace=True)
    return frame


def goods_sales_min_uv_cal(df=pd.DataFrame()):
    """计算goods 7天内最小的购买人数"""
    frame = df[['goods_id', 'sales_uv', 'data_date']].groupby('goods_id').agg(
        {'sales_uv': 'min', 'data_date': 'count'}).reset_index()
    frame.rename(columns={'sales_uv': 'goods_sales_min_uv'}, inplace=True)
    frame['goods_sales_min_uv'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['goods_sales_min_uv'], axis=1)
    return frame[['goods_id', 'goods_sales_min_uv']]


def goods_sales_avg_uv_cal(df=pd.DataFrame()):
    """计算goods 7天内平均的购买人数"""
    frame = df[['goods_id', 'sales_uv']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'sales_uv': 'sales_avg_uv'}, inplace=True)
    frame['sales_avg_uv'] = frame['sales_avg_uv'] / 7
    return frame


def goods_sales_var_uv_cal(df=pd.DataFrame()):
    """计算goods 7天内每天购买人数的方差"""
    frame = df[['goods_id', 'sales_uv']].groupby('goods_id').apply(var_value_cal).reset_index()
    frame.columns = ['goods_id', 'goods_sales_var_uv']
    return frame


goods_sales_uv_relat = [goods_sales_total_uv_cal, goods_sales_max_uv_cal,
                        goods_sales_min_uv_cal, goods_sales_avg_uv_cal, goods_sales_var_uv_cal]


# ---------------------------------------------------------------------------------------------------------#

def goods_onsales_days_cal(df=pd.DataFrame()):
    """计算goods 7天内的销售天数"""
    frame = df[['goods_id', 'onsale_days']].groupby('goods_id').agg(['max', 'min']).reset_index()
    frame['goods_sale_days'] = frame['onsale_days']['max'] - frame['onsale_days']['min']
    frame['goods_sale_days'] = frame['goods_sale_days'].add(1)
    pdd = frame[['goods_id', 'goods_sale_days']]
    pdd.columns = ['goods_id', 'goods_sale_days']
    return pdd


def add_goods_feature(df=pd.DataFrame()):
    goods_features = [goods_onsales_days_cal]
    goods_features.extend(goods_nomal_click_relat)
    goods_features.extend(goods_favor_click_relat)
    goods_features.extend(goods_cart_click_relat)
    goods_features.extend(goods_sales_uv_relat)
    goods_feature_data = df[['goods_id']].drop_duplicates()
    for fun in goods_features:
        frame = fun(df)
        goods_feature_data = pd.merge(goods_feature_data, frame, on='goods_id')

    return goods_feature_data


def date_split(date, lastTime):
    period = (lastTime - date).days
    if period <= 6:
        return True
    else:
        return False


def add_goods_lastweek_feature(df=pd.DataFrame()):
    df['date'] = pd.to_datetime(df.data_date, format='%Y%m%d')
    last_date = datetime.strptime(str(max(df.data_date)), '%Y%m%d')

    df = df[df['date'].apply(date_split, lastTime=last_date)]
    del df['date']

    goods_features = [goods_onsales_days_cal]
    goods_features.extend(goods_nomal_click_relat)
    goods_features.extend(goods_favor_click_relat)
    goods_features.extend(goods_cart_click_relat)
    goods_features.extend(goods_sales_uv_relat)
    goods_feature_data = df[['goods_id']].drop_duplicates()
    for fun in goods_features:
        frame = fun(df)
        goods_feature_data = pd.merge(goods_feature_data, frame, on='goods_id')

    return goods_feature_data


# ---------------------------------------------------------------------------------------------------------#

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

# # ---------------------------------------------------------------------------------------------------------#
#
# # feature about brands
#
# #1 for brands click
#
# def brand_total_click_cal(df=pd.DataFrame()):
#     """计算brand7天内总共的点击数"""
#     frame = df[['brand_id','goods_click']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'goods_click': 'brand_total_click'}, inplace=True)
#     return frame
#
# def brand_max_click_cal(df=pd.DataFrame()):
#     """计算brand7天内的最大点击数"""
#     frame = df[['brand_id', 'goods_click','data_date']].groupby(['brand_id','data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'goods_click']].groupby('brand_id').agg('max').reset_index()
#     frame.rename(columns={'goods_click': 'brand_max_click'}, inplace=True)
#     return frame
#
# def brand_min_click_cal(df=pd.DataFrame()):
#     """计算brand7天内的最小点击数"""
#     frame = df[['brand_id', 'goods_click','data_date']].groupby(['brand_id','data_date']).agg('sum').reset_index()
#     frame =frame[['brand_id', 'goods_click','data_date']].groupby(['brand_id']).agg({'goods_click':'min','data_date':'count'}).reset_index()
#     frame.rename(columns={'goods_click': 'brand_min_click'}, inplace=True)
#     frame['brand_min_click'] = frame.apply(lambda x:0 if x['data_date'] != 7 else x['brand_min_click'],axis=1)
#     return frame[['brand_id','brand_min_click']]
#
# def brand_avg_click_cal(df=pd.DataFrame()):
#     """计算brand7天内的平均点击数"""
#     frame = df[['brand_id', 'goods_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'goods_click']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'goods_click': 'brand_avg_click'}, inplace=True)
#     frame['brand_avg_click'] = frame['brand_avg_click'] / 7
#     return frame
#
# def brand_var_click_cal(df=pd.DataFrame()):
#     """计算brand7天内的点击数方差"""
#     frame = df[['brand_id', 'goods_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'goods_click']].groupby('brand_id').apply(var_value_cal).reset_index()
#     frame.columns = ['brand_id','brand_var_click']
#     return frame
#
# brand_nomal_click_relat = [brand_total_click_cal,brand_max_click_cal,
#                            brand_min_click_cal,brand_avg_click_cal,brand_var_click_cal]

# ----------------------------------------------------------------------------------------------------------#

# #2 for brand cart click
#
# def brand_cart_total_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内总共的添加购物车数"""
#     frame = df[['brand_id','cart_click']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'cart_click': 'brand_cart_var_click'}, inplace=True)
#     return frame
#
# def brand_cart_max_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内最大的添加购物车数"""
#     frame = df[['brand_id', 'cart_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'cart_click']].groupby('brand_id').agg('max').reset_index()
#     frame.rename(columns={'cart_click': 'brand_cart_max_click'}, inplace=True)
#     return frame
#
# def brand_cart_min_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内最小的添加购物车数"""
#     frame = df[['brand_id', 'cart_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'cart_click','data_date']].groupby('brand_id').agg({'cart_click':'min','data_date':'count'}).reset_index()
#     frame.rename(columns={'cart_click': 'brand_cart_min_click'}, inplace=True)
#     frame['brand_cart_min_click'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['brand_cart_min_click'], axis=1)
#     return frame[['brand_id','brand_cart_min_click']]
#
# def brand_cart_avg_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内平均的添加购物车数"""
#     frame = df[['brand_id', 'cart_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'cart_click']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'cart_click': 'brand_cart_avg_click'}, inplace=True)
#     frame['brand_cart_avg_click'] = frame['brand_cart_avg_click'] / 7
#     return frame
#
# def brand_cart_var_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内每天添加购物车数的方差"""
#     frame = df[['brand_id', 'cart_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'cart_click']].groupby('brand_id').apply(var_value_cal).reset_index()
#     frame.columns = ['brand_id','brand_cart_var_click']
#     return frame
#
# brand_cart_click_relat = [brand_cart_total_click_cal,brand_cart_max_click_cal,
#                           brand_cart_min_click_cal,brand_cart_avg_click_cal,brand_cart_var_click_cal]
#
# #----------------------------------------------------------------------------------------------------------#
#
# #3 for goods favorites click
#
# def brand_favor_total_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内总总收藏次数"""
#     frame = df[['brand_id', 'favorites_click']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'favorites_click': 'brand_favor_total_click'}, inplace=True)
#     return frame
#
# def brand_favor_max_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内最大的添加购物车数"""
#     frame = df[['brand_id', 'favorites_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'favorites_click']].groupby('brand_id').agg('max').reset_index()
#     frame.rename(columns={'favorites_click': 'brand_favor_max_click'}, inplace=True)
#     return frame
#
# def brand_favor_min_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内最小的添加购物车数"""
#     frame = df[['brand_id', 'favorites_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'favorites_click','data_date']].groupby('brand_id').agg({'favorites_click':'min','data_date':'count'}).reset_index()
#     frame.rename(columns={'favorites_click': 'brand_favor_min_click'}, inplace=True)
#     frame['brand_favor_min_click'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['brand_favor_min_click'], axis=1)
#     return frame[['brand_id','brand_favor_min_click']]
#
# def brand_favor_avg_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内平均的添加购物车数"""
#     frame = df[['brand_id', 'favorites_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'favorites_click']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'favorites_click': 'brand_favor_avg_click'}, inplace=True)
#     frame['brand_favor_avg_click'] = frame['brand_favor_avg_click'] / 7
#     return frame
#
# def brand_favor_var_click_cal(df=pd.DataFrame()):
#     """计算brand 7天内每天添加购物车数的方差"""
#     frame = df[['brand_id', 'favorites_click', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'favorites_click']].groupby('brand_id').apply(var_value_cal).reset_index()
#     frame.columns = ['brand_id','brand_favor_var_click']
#     return frame
#
# brand_favor_click_relat = [brand_favor_total_click_cal,brand_favor_max_click_cal,
#                            brand_favor_min_click_cal,brand_favor_avg_click_cal,brand_favor_var_click_cal]
#
# #----------------------------------------------------------------------------------------------------------#
#
# #4 for sales uv
#
# def brand_sales_total_uv_cal(df=pd.DataFrame()):
#     """计算brand 7天内总购买人数"""
#     frame = df[['brand_id', 'sales_uv']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'sales_uv': 'brand_sales_total_uv'}, inplace=True)
#     return frame
#
# def brand_sales_max_uv_cal(df=pd.DataFrame()):
#     """计算brand 7天内最大的购买人数"""
#     frame = df[['brand_id', 'sales_uv', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'sales_uv']].groupby('brand_id').agg('max').reset_index()
#     frame.rename(columns={'sales_uv': 'brand_sales_max_uv'}, inplace=True)
#     return frame
#
# def brand_sales_min_uv_cal(df=pd.DataFrame()):
#     """计算brand 7天内最小的购买人数"""
#     frame = df[['brand_id', 'sales_uv', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'sales_uv','data_date']].groupby('brand_id').agg({'sales_uv':'min','data_date':'count'}).reset_index()
#     frame.rename(columns={'sales_uv': 'brand_sales_min_uv'}, inplace=True)
#     frame['brand_sales_min_uv'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['brand_sales_min_uv'], axis=1)
#     return frame[['brand_id','brand_sales_min_uv']]
#
# def brand_sales_avg_uv_cal(df=pd.DataFrame()):
#     """计算brand 7天内平均的购买人数"""
#     frame = df[['brand_id', 'sales_uv', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'sales_uv']].groupby('brand_id').agg('sum').reset_index()
#     frame.rename(columns={'sales_uv': 'brand_sales_avg_uv'}, inplace=True)
#     frame['brand_sales_avg_uv'] = frame['brand_sales_avg_uv'] / 7
#     return frame
#
# def brand_sales_var_uv_cal(df=pd.DataFrame()):
#     """计算brand 7天内每天购买人数的方差"""
#     frame = df[['brand_id', 'sales_uv', 'data_date']].groupby(['brand_id', 'data_date']).agg('sum').reset_index()
#     frame = frame[['brand_id', 'sales_uv']].groupby('brand_id').apply(var_value_cal).reset_index()
#     frame.columns = ['brand_id','brand_sales_var_uv']
#     return frame
#
# brand_sales_uv_relat = [brand_sales_total_uv_cal,brand_sales_max_uv_cal,
#                         brand_sales_min_uv_cal,brand_sales_avg_uv_cal,brand_sales_var_uv_cal]
#
# # ---------------------------------------------------------------------------------------------------------#
#
# def brand_onsales_days_cal(df=pd.DataFrame()):
#     """计算goods 7天内的销售天数"""
#     frame = df[['brand_id', 'data_date']]
#     frame['max_date'] = df['data_date']
#     frame.rename(columns={'data_date':'min_date'},inplace=True)
#     frame = frame.groupby('brand_id').agg({'max_date':'max','min_date':'min'}).reset_index()
#     frame['brand_sale_days'] = frame.apply(lambda x:(datetime.strptime(str(x['max_date']),'%Y%m%d') - datetime.strptime(str(x['min_date']),'%Y%m%d')).days + 1,axis=1)
#     return frame[['brand_id','brand_sale_days']]
#
# def add_brand_feature(df=pd.DataFrame()):
#     brand_features = list()
#     brand_features.extend(brand_nomal_click_relat)
#     brand_features.extend(brand_favor_click_relat)
#     brand_features.extend(brand_cart_click_relat)
#     brand_features.extend(brand_sales_uv_relat)
#     brand_feature_data = df[['brand_id']].drop_duplicates()
#     for fun in brand_features:
#         frame = fun(df)
#         brand_feature_data = pd.merge(brand_feature_data,frame,on='brand_id')
#
#     return brand_feature_data

# ---------------------------------------------------------------------------------------------------------#

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

# ---------------------------------------------------------------------------------------------------------#

# feature about sales

# for sku salse num
def sku_sale_total_num_cal(df=pd.DataFrame()):
    """计算sku7天内总共的销量"""
    frame = df[['sku_id', 'goods_num']].groupby('sku_id').agg('sum').reset_index()
    frame.rename(columns={'goods_num': 'sku_sale_total_num'}, inplace=True)
    return frame


def sku_sale_max_num_cal(df=pd.DataFrame()):
    """计算sku7天内的最大销量"""
    frame = df[['sku_id', 'goods_num']].groupby('sku_id').agg('max').reset_index()
    frame.rename(columns={'goods_num': 'sku_sale_max_num'}, inplace=True)
    return frame


def sku_sale_min_num_cal(df=pd.DataFrame()):
    """计算sku7天内的最小销量"""
    frame = df[['sku_id', 'goods_num', 'data_date']].groupby('sku_id').agg(
        {'goods_num': 'min', 'data_date': 'count'}).reset_index()
    frame.rename(columns={'goods_num': 'sku_sale_min_num'}, inplace=True)
    frame['sku_sale_min_num'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['sku_sale_min_num'], axis=1)
    return frame[['sku_id', 'sku_sale_min_num']]


def sku_sale_avg_num_cal(df=pd.DataFrame()):
    """计算sku7天内的平均销量"""
    frame = df[['sku_id', 'goods_num']].groupby('sku_id').agg('sum').reset_index()
    frame.rename(columns={'goods_num': 'sku_sale_avg_num'}, inplace=True)
    frame['sku_sale_avg_num'] = frame['sku_sale_avg_num'] / 7
    return frame


def sku_sale_var_num_cal(df=pd.DataFrame()):
    """计算sku7天内的销量方差"""
    frame = df[['sku_id', 'goods_num']].groupby('sku_id').apply(var_value_cal).reset_index()
    frame.columns = ['sku_id', 'sku_sale_var_num']
    return frame


sku_sale_num_relat = [sku_sale_total_num_cal, sku_sale_max_num_cal,
                      sku_sale_min_num_cal, sku_sale_avg_num_cal, sku_sale_var_num_cal]


# ---------------------------------------------------------------------------------------------------------#

def goods_sale_total_num_cal(df=pd.DataFrame()):
    """计算goods7天内总共的销量"""
    frame = df[['goods_id', 'goods_num']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'goods_num': 'goods_sale_total_num'}, inplace=True)
    return frame


def goods_sale_max_num_cal(df=pd.DataFrame()):
    """计算goods7天内的最大销量"""
    frame = df[['goods_id', 'goods_num', 'data_date']].groupby(['goods_id', 'data_date']).agg('sum').reset_index()
    frame = frame[['goods_id', 'goods_num']].groupby('goods_id').agg('max').reset_index()
    frame.rename(columns={'goods_num': 'goods_sale_max_num'}, inplace=True)
    return frame


def goods_sale_min_num_cal(df=pd.DataFrame()):
    """计算goods7天内的最小销量"""
    frame = df[['goods_id', 'goods_num', 'data_date']].groupby(['goods_id', 'data_date']).agg('sum').reset_index()
    frame = frame[['goods_id', 'goods_num', 'data_date']].groupby(['goods_id']).agg(
        {'goods_num': 'min', 'data_date': 'count'}).reset_index()
    frame.rename(columns={'goods_num': 'goods_sale_min_num'}, inplace=True)
    frame['goods_sale_min_num'] = frame.apply(lambda x: 0 if x['data_date'] != 7 else x['goods_sale_min_num'], axis=1)
    return frame[['goods_id', 'goods_sale_min_num']]


def goods_sale_avg_num_cal(df=pd.DataFrame()):
    """计算goods7天内的平均销量"""
    frame = df[['goods_id', 'goods_num', 'data_date']].groupby(['goods_id', 'data_date']).agg('sum').reset_index()
    frame = frame[['goods_id', 'goods_num']].groupby('goods_id').agg('sum').reset_index()
    frame.rename(columns={'goods_num': 'goods_sale_avg_num'}, inplace=True)
    frame['goods_sale_avg_num'] = frame['goods_sale_avg_num'] / 7
    return frame


def goods_sale_var_num_cal(df=pd.DataFrame()):
    """计算goods7天内的销量方差"""
    frame = df[['goods_id', 'goods_num', 'data_date']].groupby(['goods_id', 'data_date']).agg('sum').reset_index()
    frame = frame[['goods_id', 'goods_num']].groupby('goods_id').apply(var_value_cal).reset_index()
    frame.columns = ['goods_id', 'goods_sale_var_num']
    return frame


goods_sale_num_relat = [goods_sale_total_num_cal, goods_sale_max_num_cal,
                        goods_sale_min_num_cal, goods_sale_avg_num_cal, goods_sale_var_num_cal]


# ---------------------------------------------------------------------------------------------------------#

# for price feature
# for sku price
def sku_sale_max_price_cal(df=pd.DataFrame()):
    """计算sku7天内的最大平均价格"""
    frame = df[['sku_id', 'goods_price']].groupby('sku_id').agg('max').reset_index()
    frame.rename(columns={'goods_price': 'sku_sale_max_price'}, inplace=True)
    return frame


def sku_sale_min_price_cal(df=pd.DataFrame()):
    """计算sku7天内的最小平均价格"""
    frame = df[['sku_id', 'goods_price']].groupby('sku_id').agg({'goods_price': 'min'}).reset_index()
    frame.rename(columns={'goods_price': 'sku_sale_min_price'}, inplace=True)
    return frame


def sku_sale_avg_price_cal(df=pd.DataFrame()):
    """计算sku7天内的平均平均价格"""
    frame = df[['sku_id', 'goods_price']].groupby('sku_id').agg('mean').reset_index()
    frame.rename(columns={'goods_price': 'sku_sale_avg_price'}, inplace=True)
    return frame


# def sku_sale_var_price_cal(df=pd.DataFrame()):
#     """计算sku7天内的平均价格方差"""
#     frame = df[['sku_id', 'goods_price']].groupby('sku_id').agg('var').reset_index()
#     frame.rename(columns={'goods_price': 'sku_sale_var_price'}, inplace=True)
#     return frame

sku_sale_price_relat = [sku_sale_max_price_cal, sku_sale_min_price_cal,
                        sku_sale_avg_price_cal]


# ---------------------------------------------------------------------------------------------------------#

def goods_sale_max_price_cal(df=pd.DataFrame()):
    """
    
    :param df: goods_id,goods_real_price
    :return: 
    """
    frame = df[['goods_id', 'goods_real_price']].groupby('goods_id').agg('max').reset_index()
    frame.rename(columns={'goods_real_price': 'goods_sale_max_price'}, inplace=True)
    return frame


def goods_sale_min_price_cal(df=pd.DataFrame()):
    """计算goods7天内的最小平均价格"""
    frame = df[['goods_id', 'goods_real_price']].groupby('goods_id').agg('min').reset_index()
    frame.rename(columns={'goods_real_price': 'goods_sale_min_price'}, inplace=True)
    return frame


def goods_sale_avg_price_cal(df=pd.DataFrame()):
    """计算goods7天内的平均平均价格"""
    frame = df[['goods_id', 'goods_real_price']].groupby('goods_id').agg('mean').reset_index()
    frame.rename(columns={'goods_real_price': 'goods_sale_avg_price'}, inplace=True)
    return frame


def goods_sale_var_price_cal(df=pd.DataFrame()):
    """计算goods7天内的平均价格方差"""
    frame = df[['goods_id', 'goods_real_price']].groupby('goods_id').agg('var').reset_index()
    frame.rename(columns={'goods_real_price': 'goods_sale_var_price'}, inplace=True)
    return frame


goods_sale_price_relat = [goods_sale_max_price_cal, goods_sale_min_price_cal,
                          goods_sale_avg_price_cal]


# ---------------------------------------------------------------------------------------------------------#

# for sku discount
# for sku price
def sku_sale_max_discount_rate_cal(df=pd.DataFrame()):
    """计算sku7天内的最大平均折扣率"""
    frame = df[['sku_id', 'goods_price', 'orginal_shop_price']]
    frame['discount_rate'] = frame.goods_price / frame.orginal_shop_price
    frame = frame[['sku_id', 'discount_rate']].groupby('sku_id').agg('max').reset_index()
    frame.discount_rate.fillna(0, inplace=True)
    frame.rename(columns={'discount_rate': 'sku_sale_max_discount_rate'}, inplace=True)
    return frame[['sku_id', 'sku_sale_max_discount_rate']]


def sku_sale_min_discount_rate_cal(df=pd.DataFrame()):
    """计算sku7天内的最小均折扣率"""
    frame = df[['sku_id', 'goods_price', 'orginal_shop_price']]
    frame['discount_rate'] = frame.goods_price / frame.orginal_shop_price
    frame = frame[['sku_id', 'discount_rate']].groupby('sku_id').agg('min').reset_index()
    frame.discount_rate.fillna(0, inplace=True)
    frame.rename(columns={'discount_rate': 'sku_sale_min_discount_rate'}, inplace=True)
    return frame[['sku_id', 'sku_sale_min_discount_rate']]


def sku_sale_max_discount_cal(df=pd.DataFrame()):
    """计算sku7天内的最大折扣额"""
    frame = df[['sku_id', 'goods_price', 'orginal_shop_price']]
    frame['discount_account'] = frame.orginal_shop_price - frame.goods_price
    frame = frame[['sku_id', 'discount_account']].groupby('sku_id').agg('max').reset_index()
    frame.discount_account.fillna(0, inplace=True)
    frame.rename(columns={'discount_account': 'sku_sale_max_discount'}, inplace=True)
    return frame[['sku_id', 'sku_sale_max_discount']]


sku_sale_discount_relat = [sku_sale_max_discount_rate_cal, sku_sale_min_discount_rate_cal,
                           sku_sale_max_discount_cal]


def add_sale_feature(df=pd.DataFrame()):
    """添加销售特征"""
    sku_sale_features = list()
    goods_sale_features = list()

    sku_sale_features.extend(sku_sale_num_relat)
    sku_sale_features.extend(sku_sale_price_relat)
    sku_sale_features.extend(sku_sale_discount_relat)

    goods_sale_features.extend(goods_sale_num_relat)
    goods_sale_features.extend(goods_sale_price_relat)

    sku_sale_feature_data = df[['sku_id']].drop_duplicates()
    for fun in sku_sale_features:
        frame = fun(df)
        sku_sale_feature_data = pd.merge(sku_sale_feature_data, frame, on='sku_id')

    # 求得goods的真实平均价格
    df['goods_sale_account'] = df.goods_price * df.goods_num
    frame = df[['goods_id', 'goods_sale_account', 'goods_num', 'data_date']].groupby(['goods_id', 'data_date']).agg(
        'sum').reset_index()
    frame['goods_real_price'] = frame.goods_sale_account / frame.goods_num
    df = pd.merge(df, frame[['goods_id', 'goods_real_price']], on='goods_id')
    goods_sale_feature_data = df[['goods_id']].drop_duplicates()
    for fun in goods_sale_features:
        frame = fun(df)
        goods_sale_feature_data = pd.merge(goods_sale_feature_data, frame, on='goods_id')

    sku_goods_relat = pd.read_csv(r'E:\PycharmProjects\supply-chain\dataset\goods_sku_relation.csv')

    sale_feature_data = pd.merge(sku_goods_relat, sku_sale_feature_data, on='sku_id')
    sale_feature_data = pd.merge(sale_feature_data, goods_sale_feature_data, on='goods_id')

    return sale_feature_data


def add_sale_lastweek_feature(df=pd.DataFrame()):
    """添加销售特征"""

    df['date'] = pd.to_datetime(df.data_date, format='%Y%m%d')
    last_date = datetime.strptime(str(max(df.data_date)), '%Y%m%d')

    df = df[df['date'].apply(date_split, lastTime=last_date)]
    del df['date']

    sku_sale_features = list()
    goods_sale_features = list()

    sku_sale_features.extend(sku_sale_num_relat)
    sku_sale_features.extend(sku_sale_price_relat)
    sku_sale_features.extend(sku_sale_discount_relat)

    goods_sale_features.extend(goods_sale_num_relat)
    goods_sale_features.extend(goods_sale_price_relat)

    sku_sale_feature_data = df[['sku_id']].drop_duplicates()
    for fun in sku_sale_features:
        frame = fun(df)
        sku_sale_feature_data = pd.merge(sku_sale_feature_data, frame, on='sku_id')

    # 求得goods的真实平均价格
    df['goods_sale_account'] = df.goods_price * df.goods_num
    frame = df[['goods_id', 'goods_sale_account', 'goods_num', 'data_date']].groupby(['goods_id', 'data_date']).agg(
        'sum').reset_index()
    frame['goods_real_price'] = frame.goods_sale_account / frame.goods_num
    df = pd.merge(df, frame[['goods_id', 'goods_real_price']], on='goods_id')
    goods_sale_feature_data = df[['goods_id']].drop_duplicates()
    for fun in goods_sale_features:
        frame = fun(df)
        goods_sale_feature_data = pd.merge(goods_sale_feature_data, frame, on='goods_id')

    sku_goods_relat = pd.read_csv(r'E:\PycharmProjects\supply-chain\dataset\goods_sku_relation.csv')

    sale_feature_data = pd.merge(sku_goods_relat, sku_sale_feature_data, on='sku_id')
    sale_feature_data = pd.merge(sale_feature_data, goods_sale_feature_data, on='goods_id')

    return sale_feature_data


# ---------------------------------------------------------------------------------------------------------#

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

# ---------------------------------------------------------------------------------------------------------#

# feature about promote

# def goods_promote_max_shop_price_cal(df=pd.DataFrame()):
#     """计算goods的最大促销价"""
#     df = df[['goods_id','promote_price']].agg('max').reset_index()
#     df.rename(columns={'promote_price':'goods_promote_max_shop_price'},inplace=True)
#
#     return df
#
#
# def goods_promote_min_shop_price_cal(df=pd.DataFrame()):
#     """计算goods的最小促销价"""
#     df = df[['goods_id', 'promote_price']].agg('min').reset_index()
#     df.rename(columns={'promote_price': 'goods_promote_min_shop_price'}, inplace=True)
#
#     return df


# def goods_promote_avg_shop_price_cal(df=pd.DataFrame()):
#     """计算goods的平均促销价"""
#     df = df[['goods_id', 'promote_price']].agg('mean').reset_index()
#     df.rename(columns={'promote_price': 'goods_promote_avg_shop_price'}, inplace=True)
#
#     return df

def goods_promote_min_discount_rate_cal(df=pd.DataFrame()):
    """计算最高促销折扣率"""
    df['discount_rate'] = df['promote_price'] / df['shop_price']
    df = df[['goods_id', 'discount_rate']].groupby('goods_id').agg('min').reset_index()
    df.rename(columns={'discount_rate': 'goods_promote_min_discount_rate'}, inplace=True)

    return df


def goods_promote_max_discout_rate_cal(df=pd.DataFrame()):
    """计算最低促销折扣率"""
    df['discount_rate'] = df.promote_price / df.shop_price
    df = df[['goods_id', 'discount_rate']].groupby('goods_id').agg('max').reset_index()
    df.rename(columns={'discount_rate': 'goods_promote_max_discount_rate'}, inplace=True)

    return df


def goods_promote_avg_discout_rate_cal(df=pd.DataFrame()):
    """计算平均促销折扣率"""
    df['discount_rate'] = df.promote_price / df.shop_price
    df = df[['goods_id', 'discount_rate']].groupby('goods_id').agg('mean').reset_index()
    df.rename(columns={'discount_rate': 'goods_promote_avg_discount_rate'}, inplace=True)

    return df


goods_promote_date_relatless = [goods_promote_min_discount_rate_cal, goods_promote_max_discout_rate_cal,
                                goods_promote_avg_discout_rate_cal]


def goods_promote_total_dates_cal(df=pd.DataFrame()):
    """计算goods 7天内的总促销天数"""
    df['goods_promote_total_dates'] = df.apply(lambda x: (x['end'] - x['start']).days + 1, axis=1)
    df = df[['goods_id', 'goods_promote_total_dates']].groupby('goods_id').agg('sum').reset_index()

    return df


def goods_promote_max_period_cal(df=pd.DataFrame()):
    """计算goods 7天内的最长促销天数"""
    df['goods_promote_max_period'] = df.apply(lambda x: (x['end'] - x['start']).days + 1, axis=1)
    df = df[['goods_id', 'goods_promote_max_period']].groupby('goods_id').agg('max').reset_index()
    return df


def goods_promote_min_period_cal(df=pd.DataFrame()):
    """计算goods 7天内的最长促销天数"""
    df['goods_promote_min_period'] = df.apply(lambda x: (x['end'] - x['start']).days + 1, axis=1)
    df = df[['goods_id', 'goods_promote_min_period']].groupby('goods_id').agg('max').reset_index()
    return df


def weekends_cal(df):
    """计算周末数量的函数"""
    return df[(df.dayofweek == 4) | (df.dayofweek == 5)].__len__()


def goods_promote_weekends_num_cal(df=pd.DataFrame()):
    """计算goods 7天内的周末总天数天数"""
    df['goods_promote_weekends_num'] = df.apply(lambda x: weekends_cal(pd.date_range(x['start'], x['end'])), axis=1)
    df = df[['goods_id', 'goods_promote_weekends_num']].groupby('goods_id').agg('sum').reset_index()

    return df


def holidays_cal(df):
    dates = [20170101, 20170424, 20170525, 20170625, 20170626, 20170627, 20170831,
             2017071, 2017072, 2017073, 20170921, 20171130, 20171201, 20171202,
             20171203, 20180101, 20180414, 20180501, 20180517, 20180602]

    dateList = pd.to_datetime(pd.Series(dates), format='%Y%m%d')

    return [i for i in range(len(dateList)) if dateList.iloc[i].date() in df.date].__len__()


def goods_promote_holidays_num_cal(df=pd.DataFrame()):
    """计算促销时间内节日的个数"""

    df['goods_promote_holidays_num'] = df.apply(lambda x: holidays_cal(pd.date_range(x['start'], x['end'])), axis=1)

    return df[['goods_id', 'goods_promote_holidays_num']]


def goods_promote_holidays_fisrtday_cal(df=pd.DataFrame()):
    """计算促销时间最后一天的星期"""
    df['goods_promote_holidays_fisrtday'] = df.start.map(lambda x: x.weekday())

    return df[['goods_id', 'goods_promote_holidays_fisrtday']]


def goods_promote_holidays_lastday_cal(df=pd.DataFrame()):
    """计算促销时间最后一天的星期"""
    df['goods_promote_holidays_lastday'] = df.end.map(lambda x: x.weekday())

    return df[['goods_id', 'goods_promote_holidays_lastday']]


goods_promote_date_relat = [goods_promote_total_dates_cal, goods_promote_max_period_cal,
                            goods_promote_min_period_cal, goods_promote_weekends_num_cal,
                            goods_promote_holidays_num_cal, goods_promote_holidays_fisrtday_cal,
                            goods_promote_holidays_lastday_cal]


def add_promote_feature(df=pd.DataFrame()):
    """添加goods的promotes特征"""
    goods_promote_feature_data = df[['goods_id']].drop_duplicates()

    for fun in goods_promote_date_relatless:
        frame = fun(df)
        goods_promote_feature_data = pd.merge(goods_promote_feature_data, frame, on='goods_id')

    df.promote_start_time = pd.to_datetime(df.promote_start_time, format='%Y-%m-%d %H:%M:%S')
    df.promote_end_time = pd.to_datetime(df.promote_end_time, format='%Y-%m-%d %H:%M:%S')
    df.start = pd.to_datetime(df.start, format='%Y-%m-%d')
    df.end = pd.to_datetime(df.end, format='%Y-%m-%d')

    df.start = df.apply(lambda x: max(x['start'], x['promote_start_time']), axis=1)
    df.end = df.apply(lambda x: min(x['promote_end_time'], x['end']), axis=1)
    df = df[['goods_id', 'start', 'end']].groupby(['goods_id', 'start']).agg('max').reset_index()
    df = df[['goods_id', 'start', 'end']].groupby(['goods_id', 'end']).agg('min').reset_index()

    for fun in goods_promote_date_relat:
        frame = fun(df)
        goods_promote_feature_data = pd.merge(goods_promote_feature_data, frame, on='goods_id')

    return goods_promote_feature_data


# ---------------------------------------------------------------------------------------------------------#

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

# ---------------------------------------------------------------------------------------------------------#

# other feature

def sku_under_same_goods_num(df=pd.DataFrame()):
    """计算一个sku有多少同属goods"""
    df = df.groupby('goods_id').agg('count').reset_index()
    df.rename(columns={'sku_id': 'sku_under_same_goods'}, inplace=True)

    return df


# def sku_total_sales_num_cal(df=pd.DataFrame()):
#     """计算每个sales的总共销量"""
#     df = df.


# feature between sku and goods
# def sku_goods_sale_max_scale_cal():


# def add_sku_goods_feature(df=pd.DataFrame()):
#     """添加sku和goods的交叉特征"""
#     df = df.


if __name__ == '__main__':

    goods_brand_relat = pd.read_csv(r'E:\PycharmProjects\supply-chain\dataset\goodsinfo.csv')
    goods_brand_relat = goods_brand_relat[['goods_id','brand_id']].drop_duplicates()

    # 得到daliy训练集关于goods的对应特征，brand特征被删去是因为缺失值太多
    Trainpath = r'E:\PycharmProjects\supply-chain\TrainSets\GoodsDailyTrainSet'
    for i in range(6):
        trainPath = Trainpath + str(i+1) + '.csv'
        df = pd.read_csv(trainPath)
        # df = pd.merge(df,goods_brand_relat,on='goods_id')
        goods = add_goods_feature(df)
        # goodspath = r'E:\PycharmProjects\supply-chain\features\goodsfeature' + str(i+1) + '.csv'
        # goods.to_csv(goodspath, index=False)
        # print('goods' + str(i+1) + 'finished.')
        # brand = add_brand_feature(df)
        # brandpath = r'E:\PycharmProjects\supply-chain\features\brandfeature' + str(i+1) + '.csv'
        # brand.to_csv(brandpath, index=False)
        # print('brand' + str(i + 1) + 'finished.')
        # tmp = pd.merge(goods, goods_brand_relat, on='goods_id')
        # final = pd.merge(tmp, brand, on='brand_id')
        finalpath = r'E:\PycharmProjects\supply-chain\features\daily7feature' + str(i+1) + '.csv'
        goods.to_csv(finalpath, index=False)
        print('feature' + str(i + 1) + 'finished.')

    Trainpath = r'E:\PycharmProjects\supply-chain\TrainSets\GoodsDailyTrainSet'
    for i in range(6):
        trainPath = Trainpath + str(i + 1) + '.csv'
        df = pd.read_csv(trainPath)
        goods = add_goods_lastweek_feature(df)
        finalpath = r'E:\PycharmProjects\supply-chain\features\daily7feature' + str(i + 1) + '.csv'
        goods.to_csv(finalpath, index=False)
        print('feature' + str(i + 1) + 'finished.')

    for i in range(6):
        trainPath = r'E:\PycharmProjects\supply-chain\TrainSets\GoodsSaleTrainSet' + str(i + 1) + '.csv'
        df = pd.read_csv(trainPath)
        frame = add_sale_lastweek_feature(df)
        sale_feature_file_name = r'E:\PycharmProjects\supply-chain\features\sale7feature' + str(i + 1) + '.csv'
        frame.to_csv(sale_feature_file_name, index=False)
        print('Feature' + str(i + 1) + 'finished!')


    #得到daliy测试集关于goods和brand的对应特征
    Testdaily = pd.read_csv(r'E:\PycharmProjects\supply-chain\TestSets\GoodsDailyTestSet.csv')
    # Testdaily = pd.merge(Testdaily, goods_brand_relat, on='goods_id')
    Testgoods = add_goods_feature(Testdaily)
    # Testbrand = add_brand_feature(Testdaily)
    # tmp = pd.merge(Testgoods,goods_brand_relat,on='goods_id')
    # final = pd.merge(tmp,Testbrand,on='brand_id')
    Testgoods.to_csv(r'E:\PycharmProjects\supply-chain\TestSets\TestDailyfeature.csv',index=False)

    # 计算销售表的关于sku和goods的sale特征
    for i in range(6):
        trainPath = r'E:\PycharmProjects\supply-chain\TrainSets\GoodsSaleTrainSet' + str(i + 1) + '.csv'
        df = pd.read_csv(trainPath)
        frame = add_sale_feature(df)
        sale_feature_file_name = r'E:\PycharmProjects\supply-chain\features\salefeature' + str(i + 1) + '.csv'
        frame.to_csv(sale_feature_file_name, index=False)
        print('Feature' + str(i + 1) + 'finished!')

    # 得到daliy测试集关于goods和brand的对应特征

    TestSale = pd.read_csv(r'E:\PycharmProjects\supply-chain\TestSets\GoodsSaleTestSet.csv')
    frame = add_sale_lastweek_feature(TestSale)
    frame.to_csv(r'E:\PycharmProjects\supply-chain\TestSets\TestSale7feature.csv', index=False)

    Testdaily = pd.read_csv(r'E:\PycharmProjects\supply-chain\TestSets\GoodsDailyTestSet.csv')
    Testgoods = add_goods_lastweek_feature(Testdaily)
    Testgoods.to_csv(r'E:\PycharmProjects\supply-chain\TestSets\TestDaily7feature.csv', index=False)

    # promote feature

    ProPath = r'E:\PycharmProjects\supply-chain\TrainSets\GoodsPromoteTrainSet'

    for i in range(6):
        propath = ProPath + str(i + 1) + '.csv'
        df = pd.read_csv(propath)
        frame = add_promote_feature(df)
        goods_feature_file_name = r'E:\PycharmProjects\supply-chain\features\promotefeature' + str(i + 1) + '.csv'
        frame.to_csv(goods_feature_file_name,index=False)
        print('Feature' +str(i + 1) + 'finished!')

    testpro = pd.read_csv(r'E:\PycharmProjects\supply-chain\TestSets\PromoteTestSet.csv')
    frame = add_promote_feature(testpro)
    goods_feature_file_name = r'E:\PycharmProjects\supply-chain\features\TestPromotefeature.csv'
    frame.to_csv(goods_feature_file_name, index=False)
