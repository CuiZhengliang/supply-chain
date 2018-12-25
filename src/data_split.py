import pandas as pd
import numpy as np
import random
from datetime import datetime
from datetime import date

if __name__ == '__main__':

    # DataPath = r'E:\PycharmProjects\Task2Plus\TrainSets'
    #
    # # for goods daily
    # goodsdaily = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goodsdaily.csv')
    #
    # trainset1 = goodsdaily[(goodsdaily.data_date >= 20170315) & (goodsdaily.data_date <= 20170612)]
    # trainset2 = goodsdaily[(goodsdaily.data_date >= 20170322) & (goodsdaily.data_date <= 20170619)]
    # trainset3 = goodsdaily[(goodsdaily.data_date >= 20170329) & (goodsdaily.data_date <= 20170626)]
    # trainset4 = goodsdaily[(goodsdaily.data_date >= 20170405) & (goodsdaily.data_date <= 20170703)]
    # trainset5 = goodsdaily[(goodsdaily.data_date >= 20170412) & (goodsdaily.data_date <= 20170710)]
    # trainset6 = goodsdaily[(goodsdaily.data_date >= 20170419) & (goodsdaily.data_date <= 20170717)]
    #
    # trainsets = [trainset1,trainset2,trainset3,trainset4,trainset5,trainset6]
    #
    # for i in range(len(trainsets)):
    #     file_name = DataPath + '\GoodsDailyTrainSet' + str(i+1) + '.csv'
    #     trainsets[i].to_csv(file_name,index=False)
    #
    # GoodsDailyTestSet = goodsdaily[(goodsdaily.data_date >= 20171217) & (goodsdaily.data_date <= 20180316)]
    # GoodsDailyTestSet.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\GoodsDailyTestSet.csv',index=False)

    # #for goods sale
    # goodsale = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goodsalePlus.csv')
    #
    # trainset1 = goodsale[(goodsale.data_date >= 20170315) & (goodsale.data_date <= 20170612)]
    # trainset2 = goodsale[(goodsale.data_date >= 20170322) & (goodsale.data_date <= 20170619)]
    # trainset3 = goodsale[(goodsale.data_date >= 20170329) & (goodsale.data_date <= 20170626)]
    # trainset4 = goodsale[(goodsale.data_date >= 20170405) & (goodsale.data_date <= 20170703)]
    # trainset5 = goodsale[(goodsale.data_date >= 20170412) & (goodsale.data_date <= 20170710)]
    # trainset6 = goodsale[(goodsale.data_date >= 20170419) & (goodsale.data_date <= 20170717)]
    #
    # trainsets = [trainset1,trainset2,trainset3,trainset4,trainset5,trainset6]
    #
    # for i in range(len(trainsets)):
    #     file_name = DataPath + '\GoodsSaleTrainSet' + str(i+1) + '.csv'
    #     trainsets[i].to_csv(file_name,index=False)
    #
    #
    # GoodsSaleTestSet = goodsale[(goodsale.data_date >= 20171217) & (goodsale.data_date <= 20180316)]
    # GoodsSaleTestSet.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\GoodsSaleTestSet.csv',index=False)

    # for market plan
    # market = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\marketing.csv')
    # trainset1 = market[(market.data_date >= 20170503) & (market.data_date <= 20170731)]
    # trainset2 = market[(market.data_date >= 20170510) & (market.data_date <= 20170807)]
    # trainset3 = market[(market.data_date >= 20170517) & (market.data_date <= 20170814)]
    # trainset4 = market[(market.data_date >= 20170524) & (market.data_date <= 20170821)]
    # trainset5 = market[(market.data_date >= 20170801) & (market.data_date <= 20171019)]
    # trainset6 = market[(market.data_date >= 20170808) & (market.data_date <= 20171026)]
    # trainset7 = market[(market.data_date >= 20170815) & (market.data_date <= 20171102)]
    # trainset8 = market[(market.data_date >= 20170822) & (market.data_date <= 20171109)]
    #
    # trainsets = [trainset1,trainset2,trainset3,trainset4,trainset5,trainset6,trainset7,trainset8]
    #
    # for i in range(len(trainsets)):
    #     file_name = DataPath + '\marketTrainSet' + str(i+1) + '.csv'
    #     trainsets[i].to_csv(file_name,index=False)
    #
    # MarketTestSet = market[(market.data_date >= 20171217) & (market.data_date <= 20180316)]
    # MarketTestSet.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\MarketTestSet.csv',index=False)

    # def judge_peroid(start,end,date1,date2):
    #     if start < date1 < end or start < date2 < end:
    #         return True
    #     else:
    #         return False
    #
    #
    # #for promote
    # promote = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\target_promote.csv')
    # promote.promote_start_time = pd.to_datetime(promote.promote_start_time,format='%Y-%m-%d %H:%M:%S')
    # promote.promote_end_time = pd.to_datetime(promote.promote_end_time,format='%Y-%m-%d %H:%M:%S')
    #
    # trainset1 = promote[promote.apply(
    #     lambda x: judge_peroid(x['promote_start_time'], x['promote_end_time'], datetime(2017, 3, 15),
    #                            datetime(2017, 6, 12)), axis=1)]
    # trainset1['start'] = datetime(2017, 3, 15)
    # trainset1['end'] = datetime(2017, 6, 12)
    #
    # trainset2 = promote[promote.apply(
    #     lambda x: judge_peroid(x['promote_start_time'], x['promote_end_time'], datetime(2017, 3, 22),
    #                            datetime(2017, 6, 19)), axis=1)]
    # trainset2['start'] = datetime(2017, 3, 22)
    # trainset2['end'] = datetime(2017, 6, 19)
    #
    # trainset3 = promote[promote.apply(
    #     lambda x: judge_peroid(x['promote_start_time'], x['promote_end_time'], datetime(2017, 3, 29),
    #                            datetime(2017, 6, 26)), axis=1)]
    # trainset3['start'] = datetime(2017, 3, 29)
    # trainset3['end'] = datetime(2017, 6, 26)
    #
    # trainset4 = promote[promote.apply(
    #     lambda x: judge_peroid(x['promote_start_time'], x['promote_end_time'], datetime(2017, 4, 5),
    #                            datetime(2017, 7, 3)), axis=1)]
    # trainset4['start'] = datetime(2017, 4, 5)
    # trainset4['end'] = datetime(2017, 7, 3)
    #
    # trainset5 = promote[promote.apply(
    #     lambda x: judge_peroid(x['promote_start_time'], x['promote_end_time'], datetime(2017, 4, 12),
    #                            datetime(2017, 7, 10)), axis=1)]
    # trainset5['start'] = datetime(2017, 4, 12)
    # trainset5['end'] = datetime(2017, 7, 10)
    #
    # trainset6 = promote[promote.apply(
    #     lambda x: judge_peroid(x['promote_start_time'], x['promote_end_time'], datetime(2017, 4, 19),
    #                            datetime(2017, 7, 17)), axis=1)]
    # trainset6['start'] = datetime(2017, 4, 19)
    # trainset6['end'] = datetime(2017, 7, 17)
    #
    # trainsets = [trainset1,trainset2,trainset3,trainset4,trainset5,trainset6]
    #
    # for i in range(6):
    #     file_name = r'E:\PycharmProjects\Task2Plus\TrainSets\GoodsPromoteTrainSet' + str(i+1) + '.csv'
    #     trainsets[i].to_csv(file_name,index=False)
    #
    # promoteTestSet = promote[promote.apply(
    #     lambda x: judge_peroid(x['promote_start_time'], x['promote_end_time'], datetime(2017, 12, 17),
    #                            datetime(2018, 3,16)), axis=1)]
    # promoteTestSet['start'] = datetime(2017, 12, 17)
    # promoteTestSet['end'] = datetime(2018, 3,16)
    # promoteTestSet.to_csv(r'E:\PycharmProjects\Task2Plus\TestSets\PromoteTestSet.csv',index=False)

    # 标签提取区间
    goodsale = pd.read_csv(r'E:\PycharmProjects\Task2Plus\dataset\goodsalePlus.csv')
    labelset1 = goodsale[(goodsale.data_date >= 20170728) & (goodsale.data_date <= 20170831)]
    labelset2 = goodsale[(goodsale.data_date >= 20170804) & (goodsale.data_date <= 20170907)]
    labelset3 = goodsale[(goodsale.data_date >= 20170811) & (goodsale.data_date <= 20170914)]
    labelset4 = goodsale[(goodsale.data_date >= 20170818) & (goodsale.data_date <= 20170921)]
    labelset5 = goodsale[(goodsale.data_date >= 20170825) & (goodsale.data_date <= 20170928)]
    labelset6 = goodsale[(goodsale.data_date >= 20170901) & (goodsale.data_date <= 20171005)]

    labelsets = [labelset1, labelset2, labelset3, labelset4, labelset5, labelset6]

    for i in range(len(labelsets)):
        file_name = r'E:\PycharmProjects\Task2Plus\LabelSets\labelset' + str(i + 1) + '.csv'
        labelsets[i].to_csv(file_name, index=False)
