from sklearn.preprocessing import OneHotEncoder

from code.feature.func import *

tools = pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\initial_feature_selection\tools.csv')
data_num.=pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\initial_feature_selection\data_num.csv')

#tools稀有类归并
tools=rare_cat_feature_combination(tools)

#kind类以下的稀有类归并
data_num,cat_list=rare_cat_feature_combination(data_num,kind=5,cat_list=True)

data_cat=pd.concat([tools,data_num.loc[:,cat_list]],axis=1)
oh= OneHotEncoder()
oh_df=pd.DataFrame(oh.fit_transform(data_cat.values).toarray(),index=data_cat.index)

#主成分只占3种一下的列
#分布差异过大列的提取