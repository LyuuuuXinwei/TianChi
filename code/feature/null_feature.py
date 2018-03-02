from code.feature.func import *

data_num = pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\data_num.csv')



#全体空值率
data_null_count, feature_null_count=get_null_count(data_num)
data_null_count.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\构造特征\data_null_count.csv')
feature_null_count.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\构造特征\feature_null_count.csv')


#单列空值特征
null_f=create_null_feature(data_num)


#样本空值率与分块空值率得到空值率与权重的关系
data_null_count['null_count_all']