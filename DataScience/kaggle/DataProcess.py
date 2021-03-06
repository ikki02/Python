# coding: utf-8
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG
from LoadData import load_train_data, load_test_data, load_suppliment
import numpy as np
import pandas as pd 
from sklearn import preprocessing

# ログの出力名を設定
logger = getLogger(__name__)

# ログのフォーマットを設定（詳細は次のマークダウンテーブル参考）
fmt = Formatter('%(asctime)s %(name)s %(lineno)d %(levelname)s %(message)s')

# ログのコンソール出力の設定
shandler = StreamHandler()
shandler.setLevel('INFO')
shandler.setFormatter(fmt)

# ログのファイル出力先の設定
fhandler = FileHandler('result_tmp/Input.log')
fhandler.setLevel(DEBUG)
fhandler.setFormatter(fmt)

# ログレベルの設定
logger.setLevel(DEBUG)
logger.addHandler(shandler)
logger.addHandler(fhandler)
logger.propagate = False


def make_train_in_test():
	logger.info('train in test starts')
	train = load_train_data()
	test = load_test_data()
	logger.info('train.org.shape:{}'.format(train.shape))
	test_shops = test.shop_id.unique()
	test_items = test.item_id.unique()
	train = train[train.shop_id.isin(test_shops)]
	train = train[train.item_id.isin(test_items)]
	train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
	train['month'] = train['date'].dt.month  #TimeSeries。月のみの抽出。
	logger.info('train in test.shape:{}'.format(train.shape))
	logger.debug('train in test ends')
	return train, test


def make_item_cnt_month(train):
	logger.info('MakeItemCntMonth starts')
    item_cnt_month = train['item_cnt_day'].groupby( \
		[train['month'], train['shop_id'], train['item_id']]).apply(sum)  #applyは任意の関数可
	item_cnt_month.name = 'item_cnt_month'
	item_cnt_month_df = pd.DataFrame(item_cnt_month)
	item_cnt_month_df = item_cnt_month_df.reset_index()  #MultiIndexの解除。カラム化。
	item_cnt_month_df.drop(['date_block_num'], axis=1, inplace=True)
	item_cnt_month_df.to_csv('./result_tmp/scaled_train.csv', encoding='utf-8-sig', index=False)
	logger.debug(item_cnt_month_df.shape)
	logger.debug('MakeItemCntMonth ends')
	return item_cnt_month_df


def label_encode(train, test):
	# 木系のアルゴリズムは使う必要なし。
	logger.info('label_encode starts')
	for c in ['shop_id', 'item_id']:
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(train[c].unique())+list(test[c].unique()))
		train[c] = lbl.transform(train[c].astype(str))
		test[c] = lbl.transform(test[c].astype(str))
		logger.debug(c)
	train.to_csv('./result_tmp/train_lbl.csv')
	test.to_csv('./result_tmp/test_lbl.csv')
	logger.debug('label encode ends')
	return train, test


if __name__ == '__main__':
	train, test = make_train_in_test()
	scaled_train = make_item_cnt_month(train)
	#train, test = label_encode(scaled_train, test)
