# coding: utf-8
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG
from LoadData import load_train_data, load_test_data, load_submission
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, TimeSeriesSplit, GroupKFold
from sklearn.ensemble import RandomForestRegressor

# ログの出力名を設定
logger = getLogger(__name__)

# ログのフォーマットを設定（詳細は次のマークダウンテーブル参考）
fmt = Formatter('%(asctime)s %(name)s %(lineno)d %(levelname)s %(message)s')

# ログのコンソール出力の設定
shandler = StreamHandler()
shandler.setLevel('INFO')
shandler.setFormatter(fmt)

# ログのファイル出力先の設定
fhandler = FileHandler('result_tmp/Scoring.log')
fhandler.setLevel(DEBUG)
fhandler.setFormatter(fmt)

# ログレベルの設定
logger.setLevel(DEBUG)
logger.addHandler(shandler)
logger.addHandler(fhandler)
logger.propagate = False


def forest_submit():
	logger.info('RandomForestRegressor start')
	logger.debug('make_train_data start')
	#train = pd.read_csv('./result_tmp/scaled_train.csv')
	train = pd.read_csv('./result_tmp/scaled_train_DateBlockNum.csv')
	#train = train[train['date_block_num']==33]  #直近1ヶ月
	train = train.loc[(30<train['date_block_num'])&(train['date_block_num']<=33)]  #直近3m
	
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month', 'date_block_num'], axis=1).values
	#X = train.drop(['item_cnt_month'], axis=1).values
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
	logger.debug('make_train_data end')

	logger.info('Fitting start')
	forest = RandomForestRegressor(n_estimators=50, random_state=1)
	forest.fit(X, y)
	logger.debug('Fitting end')
	
	logger.info('Scoring start')
	#logger.info('Accuracy on test set: {:.3f}'.format(.score(X_test, y_test)))
	test_data = load_test_data()
	test = test_data.drop(['ID'], axis=1).values
	
	submission = load_submission()
	submission['item_cnt_month'] = forest.predict(test).astype(np.float16).clip(0., 20.)
	#submission.to_csv('./result_tmp/submit_180826_1st.csv', encoding='utf-8-sig', index=False)
	submission.to_csv('./result_tmp/submit_180827_31-33.csv', encoding='utf-8-sig', index=False)
	logger.info('submission:\n{}'.format(submission.head()))
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')


# CV: KFold, StratifiedKFold, GroupKFold
# [http://scikit-learn.org/stable/modules/cross_validation.html]
def forest_cv():
	logger.info('RandomForestRegressor start')

	logger.debug('make_train_data start')
	train = pd.read_csv('./result_tmp/scaled_train.csv')
	#train = pd.read_csv('./result_tmp/scaled_train_DateBlockNum.csv')
	#train = train[train['date_block_num']==33]  #直近1ヶ月
	#train = train.loc[(30<train['date_block_num'])&(train['date_block_num']<=33)]  #直近3m
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month'], axis=1).values
	#X = train.drop(['item_cnt_month', 'date_block_num'], axis=1).values
	logger.debug('make_train_data end')

	logger.info('Cross-validation start')
	forest = RandomForestRegressor(n_estimators=50, random_state=1)
	
	#cvはKFoldのshuffle引数をTrue/Falseから選べる。
	#ただのkfoldしたいときは上記引数をFalseにすればよい。その際、インデックス順にk分割する。
	#shuffle_cvしたいときは上記引数をTrueにすればよい。毎回分割が変わる。
	kfold = KFold(n_splits=3, shuffle=True, random_state=0)
	#skf = StratifiedKFold(n_splits=3)  #skfはkfoldと同じ引数をもつ。
	scores = cross_val_score(forest, X, y, cv=kfold)
	#以下、GroupKFoldを使うときの書き方
	#groups = list(train['date_block_num'])
	#scores = cross_val_score(forest, X, y, groups, cv=GroupKFold(n_splits=3))
	logger.info('Cross-validation scores_forest: {}'.format(scores))
	logger.info('Average Cross-validation score_forest: {}'.format(scores.mean()))
	logger.debug('Cross-validation end')
	
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')
	

# TimeSeriesSplit()によるCV
# [http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html]
# 以下のイメージで分割する。TESTを予測するために、直近のTRAINを使うイメージがtscv
# TRAIN: [0] TEST: [1], TRAIN: [0 1] TEST: [2], TRAIN: [0 1 2] TEST: [3]
def forest_tscv():
	logger.info('RandomForestRegressor start')

	logger.debug('make_train_data start')
	train = pd.read_csv('./result_tmp/scaled_train.csv')
	y = train['item_cnt_month']
	X = train.drop(['item_cnt_month'], axis=1).values
	logger.debug('make_train_data end')

	logger.info('TimeSeries Cross-validation start')
	forest = RandomForestRegressor(n_estimators=50, random_state=1)
	
	tscv = TimeSeriesSplit(n_splits=3)
	scores = cross_val_score(forest, X, y, cv=tscv)
	logger.info('TSCross-validation scores_forest: {}'.format(scores))
	logger.info('Average TSCross-validation score_forest: {}'.format(scores.mean()))
	logger.debug('TimeSeries Cross-validation end')
	
	logger.debug('RandomForestRegressor end')
	logger.debug('====================')


if __name__ == '__main__':
	forest_submit()
	#forest_cv()
	#forest_tscv()