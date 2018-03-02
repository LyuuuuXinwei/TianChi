import numpy as np
import pandas as pd
import os

if not os.path.exists(r'C:\Python\kaggle\@ON-工业AI\data\raw_data'):
    os.makedirs(r'C:\Python\kaggle\@ON-工业AI\data\raw_data')

train_file=r'C:\Python\kaggle\@ON-工业AI\data\训练.xlsx'
testa_file=r'C:\Python\kaggle\@ON-工业AI\data\测试A.xlsx'
testb_file=r'C:\Python\kaggle\@ON-工业AI\data\测试B.xlsx'
train=pd.read_excel(train_file)
test_a=pd.read_excel(testa_file)
test_b=pd.read_excel(testb_file)

label=train['Y']
train.drop(['Y'],axis=1,inplace=True)
test=pd.concat([test_a,test_b])
data=pd.concat([train,test])

data_cat=data.loc[:,data.dtypes=='object']
data_num=data.loc[:,data.dtypes!='object']

tools=data_num.iloc[:,data_num.columns.str.lower().str.contains('tool')].copy()
tools_columns=tools.columns
tools=pd.concat([data_cat,tools],axis=1)
data_num.drop(tools_columns,axis=1,inplace=True)

label.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\label.csv')
tools.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\tools.csv')
data_num.to_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\data_num.csv')