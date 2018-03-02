from code.feature.func import *

data_num = pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\initial_feature_selection\data_num.csv')
#TODO:换输入

#主成分只占3种一下的列
nless_list=get_nless_feature(data_num)
cat_like_data=data_num.loc[:,nless_list].copy()

#分布差异过大列的提取
segmented_feature,outlier_feature_list=get_discrete_feature(data_num)
segmented_feature_list,outlier_feature_list=ger_max_diff_info(data_num)
data_num_normal_dis=data_num.drop(segmented_feature_list+outlier_feature_list,axis=1)

#分布差异过大的列的聚类
