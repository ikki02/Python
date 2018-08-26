import pandas as pd

# TimeSeries。datetime型への変更。インデックス化。
train['date'] = pd.to_datetime(train['date'])
train = train.set_index('date')

# 欠損値を埋めつつ結合する効率のよい書き方。
train = pd.merge(train, shop_item, how='left', on=['shop_id','item_id']).fillna(0.)

# 変数の値をファイルに記録したいときの書き方。
with open('hoge.txt', 'wt') as fout:
    print(forest.predict(test), file=fout)