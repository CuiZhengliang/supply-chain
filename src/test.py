#*encoding = UTF8
import pandas as pd

if __name__ == '__main__':
    sales = pd.read_csv('E:\PycharmProjects\Task2Plus\dataset\goodsalePlus.csv')
    relation = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goods_sku_relation.csv')

    tmp = sales[['goods_id','sku_id']].drop_duplicates()
    tmp = tmp.groupby('goods_id').agg('count').reset_index()
    tmp.rename(columns={'sku_id': 'sku_under_same_goods_num'}, inplace=True)

    relation = pd.merge(relation,tmp,on='goods_id')

    goodsale = sales[['goods_id','goods_num']].groupby('goods_id').agg('sum').reset_index()
    goodsale.rename(columns={'goods_num':'goods_total_sale'},inplace=True)
    skusale = sales[['sku_id','goods_num']].groupby('sku_id').agg('sum').reset_index()
    skusale.rename(columns={'goods_num': 'sku_total_sale'}, inplace=True)
    df = pd.merge(relation,goodsale,on='goods_id')
    df = pd.merge(df,skusale,on='sku_id')
    del df['goods_id']
    df.to_csv(r'E:\PycharmProjects\Task2Plus\dataset\totalsale.csv',index=False)

