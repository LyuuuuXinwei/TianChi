import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#initial_feature_selection
def zero_to_null(data,special_df=False):
    df=data.copy()
    df.loc[:,(df.dtypes=='float64') & (df.isnull().sum()==0)].replace(0,np.nan,inplace=True)
    special_float=df.loc[:, (df.dtypes == 'float64') & (df.isnull().sum() != 0)].copy()
    special_int=df.loc[:, (df.dtypes == 'int64') & (df.isnull().sum() != 0)].copy()
    if special_df==False:
        return df
    if special_df==True:
        return special_float,special_int


def get_data_sliced(df):
    last = ''.join([df.columns[0].split('X')[0],'X'])
    sliced = 0
    slice_list = [0,]
    name_list = [last]
    for i in df.columns:
        label = ''.join([i.split('X')[0],'X'])
        sliced += 1
        if label != last:
            slice_list.append(sliced-1)
            last = label
            name_list.append(last)
    slice_list.append(len(df.columns))

    return name_list, slice_list


def get_majority(df,threshold=0.95):
    delete_list=[]
    for i in df.columns:
        cat=df.value_counts().values
        if cat.max()>(len(df)*threshold):
            delete_list.append(i)
    return delete_list


def get_low_variance_feature(df,threshold=0.02):
    delete_list=[]
    for i in df.columns:
        std=df[i].std()
        if std<threshold:
            delete_list.append(i)
    return delete_list


def rare_value_to_median(data,threshold=5):
    df=data.copy()
    j = 0
    for i in df.columns:
        cat = df[i].value_counts().to_frame('num').reset_index()
        if len(cat) <= 10:
            exceptional_value = cat[cat['num'] < threshold]['index'].values
            if len(exceptional_value) != 0:
                j = j + 1
                df[i].replace(exceptional_value, df[i].median(), inplace=True)
    print('共修正{}列数据的异常值'.format(j))
    return df


# 删除表示时间的特征
def find_time_feature(data):
    df = data.fillna(0).copy()
    numerical_feature_median = pd.Series(np.zeros(len(df.columns)), index=df.columns)
    for i in df.columns:
        numerical_feature_median[i] = df[i].median()
    time_feature = numerical_feature_median[
        numerical_feature_median.astype('str').str.contains('2017') | numerical_feature_median.astype(
            'str').str.contains('2.017')]
    return time_feature.index



#全体与分块空缺统计
def get_null_count(data):
    data_null_count = pd.DataFrame(index=data.index)
    data_null_count['null_count_all'] = data.isnull().sum(axis=1).values

    name_list, slice_list = get_data_sliced(data)
    for i in range(len(slice_list) - 1):
        data_null_count['null_count_{}'.format(name_list[i])] = data.iloc[:,
                                        slice_list[i]:(slice_list[i + 1] - 1)].isnull().sum(axis=1).values
    feature_null_count = data.isnull().sum(axis=0)

    return data_null_count, feature_null_count



#合并高相似度的特征
def merge_similar_feature(data,threshold=0.98):
    df=data.copy()
    for i in df.columns:
        for j in df.columns:
            if i in df.columns:
                if j in df.columns:
                    if i != j:
                        if (df[i].corr(df[j])) > threshold:
                            if df[i].notnull().sum()>df[j].notnull().sum():
                                df.drop([j], axis=1, inplace=True)
                            else:
                                df.drop([i], axis=1, inplace=True)
    return df


def encode_tools(data):
    le = LabelEncoder()
    df=data.copy()
    for i in df.columns:
        df[i]=le.fit_transform(df[i])
    return df


def tools_related_feature(df,tools,corr_threshold=0.8,return_method='list'):
    dic_high_corr={}
    dic_grouped_median={}
    list_grouped_median=[]
    list_high_corr=[]
    for i in tools.columns:
        l_median=[]
        l_corr=[]
        for j in df.columns:
            tmp = pd.concat((tools[i], df[j]), axis=1)
            if len(tmp[j]) < (len(tmp[i]+10)):
                if tmp.groupby([i])[j].std().medain()==0:
                    l_median.append(j)
                    list_grouped_median.append(j)
                if abs(tmp[i].corr(tmp[j]))>corr_threshold:
                    l_corr.append(j)
                    list_high_corr.append(j)
        dic_grouped_median[i]=l_median
        dic_high_corr[i]=l_corr
    if return_method=='list':
        return list_grouped_median,list_high_corr
    else:
        return dic_grouped_median,dic_high_corr


def delete_tools_related_feature(df,tools,corr_threshold=0.8,method='union'):
    data=df.copy()
    list_grouped_median, list_high_corr = tools_related_feature(df, tools,corr_threshold=corr_threshold,return_method='list')
    intersection = set(list_grouped_median) & set(list_high_corr)
    union = set(list_grouped_median) | set(list_high_corr)
    if method=='union':
        data.drop(union,axis=1,inplace=True)
        return union
    else:
        data.drop(intersection, axis=1, inplace=True)
        return intersection



# discrete_feature_selection
#主成分只占3种一下的列
def get_nless_feature(df,kind=3,threshold=0.98):
    nless_list=[]
    for i in df.columns:
        vc=df.value_counts()
        if vc[:kind].cumsum()/len(df.columns)>threshold:
            nless_list.append(i)
    return nless_list


#分布差异过大列的提取
def get_max_diff_info(df,cat_threshold=15,max_ratio=0.15,loc=20):
    segmented_feature_list=[]
    outlier_feature_list=[]
    for i in df.columns:
        if len(df[i].unique())>cat_threshold:
            diff = np.diff(np.sort(df[i].values))
            criteria= (np.percentile(df[i],98) - np.percentile(df[i],2))*max_ratio
            if np.max(diff) > criteria:
                criteria_loc=abs(np.argmax(diff)-len(df.columns)/2)>(len(df.columns)/2-loc)
                if criteria_loc:
                    outlier_feature_list.append(i)
                else:
                    segmented_feature_list.append(i)

    return segmented_feature_list,outlier_feature_list


def get_discrete_feature(df):
    segmented_feature_list, outlier_feature_list=ger_max_diff_info(df)
    segmented_feature=df.loc[:,segmented_feature_list]
    outlier_feature_list = df.loc[:, outlier_feature_list]
    return segmented_feature,outlier_feature_list


#outlier
#小类归并
def rare_cat_feature_combination(data,kind=10,threshold=5,rare_cat_feture_name=99,cat_list=False):
    cat_list=[]
    df=data.copy()
    for i in df.columns:
        if len(df[i].unique())<kind:
            cat_list.append(i)
            cat_count=df[i].value_counts()
            rare_cat=cat_count[cat_count<=threshold].index
            if len(rare_cat)==1:
                df.loc[df[i].map(lambda x:x in rare_cat),i]=cat_count.max()
            if len(rare_cat)>1:
                df.loc[df[i].map(lambda x: x in rare_cat), i] = rare_cat_feture_name

    if cat_list==False:
        return df
    if cat_list==True:
        return df,cat_list


#大于20个空值制造空值特征
def create_null_feature(df,threshold=20):
    null_f = pd.DataFrame(index=df.index)
    for i in df.columns:
        if df[i].isnull().sum()>threshold:
            null_f['%s_isnull' % i] = df[i].isnull().astype('int').values
    return null_f


#聚类离散化
#def discretization(df):

