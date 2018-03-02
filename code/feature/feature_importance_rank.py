import numpy as np
import pandas as pd

#相关度从小到大排序
def get_corr_rank(df,label):
    cor=abs(df.loc[:499,:].corrwith(label)).order()
    #TODO:这个排序写达不到
    return cor.index



if __name__=='__main__':
#TODO:特征重要度：原始排序，筛选排序，
    get_corr_rank