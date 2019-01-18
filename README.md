# supply-chain
供应链需求预测

环境:Windows10,python==3.6.5,lightgbm==2.1.2,xgboost==0.81

* sedwork.py，数据预处理，清洗某些数字符号错误

* data_split.py，划窗得到训练集测试集

* label_biu.py，打标得到标签区间sku的每周销量

* feature_extract.py，提取特征区间的特征

* lgb_work.py，生成lgb模型并对测试集进行预测，预测未来5周的销量

* run_all.py，一键运行，得出最终结果
