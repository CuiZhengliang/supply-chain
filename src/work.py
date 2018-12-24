import pandas as pd

if __name__ == '__main__':
#SKb90aP4
    df = lgb.copy()
    last = pd.read_csv(r'E:\PycharmProjects\Task2Plus\output\PreByLastSame.csv')
    avg = pd.read_csv(r'E:\PycharmProjects\Task2Plus\output\PreByAvg.csv')
    df = df.append({'sku_id':'SKb90aP4','week1':0,'week2':0,'week3':0,'week4':0,'week5':0},ignore_index=True)
    df = pd.merge(avg[['sku_id']],df,on='sku_id')
    for j in range(5):
        title = 'week' + str(j + 1)
        df[title] = lgb1[title] * 0.2 + lgb2[title] * 0.4 + xgb1[title] * 0.2 + xgb2[title] * 0.2

    for j in range(5):
        title = 'week' + str(j + 1)
        df[title] = df[title].astype('int')

    for j in range(5):
        title = 'week' + str(j + 1)
        df[title] = df[title].map(lambda x:x if x > 0 else 0)