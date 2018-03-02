import numpy as np
import pandas as pd
import xgboost as xgb


train_y=pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\label.csv')
tools=pd.read_csv(r'C:\Python\kaggle\@ON-工业AI\data\raw_data\tools.csv')
train_x=tools[:500,:].values
test_x=tools[500:,:].values

def pipeline():
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    # dtrain.save_binary("train.buffer")
    # dtest = xgb.DMatrix('test.svm.buffer')

	params={
	    	'booster':'gbtree',
	    	'objective': 'reg:linear',
		    'eval_metric': 'rmse',
	    	'max_depth':7,
	    	'lambda':100,
		    'subsample':0.7,
		    'colsample_bytree':0.7,
		    'eta': 0.008,
	    	'seed':1024,
	    	'nthread':8
		}

	#train
	watchlist  = [(dtrain,'train')]
    #evallist = [(dtest, 'eval'), (dtrain, 'train')]
	model = xgb.train(params,dtrain,num_boost_round=500,evals=watchlist)


	#predict test set
	test_a_b['pred'] = model.predict(dtest)
    test_a_b.to_csv('test_1.csv',index=None)

    #bst.save_model('0001.model')

# bst.dump_model('dump.raw.txt')
# 转储模型和特征映射
# bst.dump_model('dump.raw.txt','featmap.txt')
# bst = xgb.Booster({'nthread':4}) #init model
# bst.load_model("model.bin") # load data

if __name__ == "__main__":
    pipeline()
