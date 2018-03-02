from code.feature.func import *

data_num = pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\data_num.csv',index_col='Unnamed: 0')
tools= pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\tools.csv')

#删除空缺率条目大于100的特征
data_null_count, feature_null_count=get_null_count(data_num)
data_num=data_num.loc[:,feature_null_count<100]

#删除标准差低于0.02的特征
delete_list=get_low_variance_feature(data_num)
low_variance_feature=pd.DataFrame(index=data_num.index)
for i in delete_list:
    low_variance_feature[i]=data_num[i].values
data_num.drop(delete_list,axis=1,inplace=True)
low_variance_feature.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\删除特征\low_variance_feature.csv')

#删除主成分大于0.98的特征
delete_list=get_majority(data_num)
data_num.drop(delete_list,axis=1,inplace=True)

#删除时间特征
time_feature = find_time_feature(data_num)
time=pd.DataFrame(index=data_num.index)
for i in time_feature:
    time[i]=data_num[i].values
data_num.drop(time_feature,axis=1,inplace=True)
time.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\删除特征\time.csv')

#数据清洗
#低类数据异常值去除
#data_num=rare_value_to_median(data_num)

#删除互相相似的特征
data_num=merge_similar_feature(data_num)

#tools编码
tools=encode_tools(tools)

#tools相关性
tools.drop(['TOOL_ID (#3)'],axis=1,inplace=True)

#删除和tools相关性过大的列
data_num=delete_tools_related_feature(data_num,tools)



tools.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\initial_feature_selection\tools.csv')
data_num.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\initial_feature_selection\data_num.csv')
