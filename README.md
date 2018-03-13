# TianChi
阿里天池工业AI大赛-智能制造质量预测  

竞赛链接：https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165261.5678.1.18356562lNgrPr&raceId=231633  


基于500条液晶显示器样本在13个工序上8029个维度的匿名生产数据，预测某一和产品质量相关的匿名目标指标。
使用filter,wrapper,embedded等不同特征选择方式分别获得原始特征的重要度排序并加权融合，得到一个“集成特征筛选器”作为降维标准；将特征保留数目作为模型的超参数，分别使用lasso,ridge,rf,svr,GBDT,等模型并bagging融合模型，增强鲁棒性；使用嵌套交叉验证+留一法LOOCV获得模型的无偏估计。


