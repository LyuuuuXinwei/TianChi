from code.feature.func import *

data_num = pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\initial_feature_selection\data_num.csv')

#空值填补
data_num.fillna()

#离群点

#0变空
data_num=zero_to_null(data_num)