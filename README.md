# Python

### modules
- sys [https://docs.python.jp/3/library/sys.html]
- .path.append('../my_modules') #モジュールを検索するパスを示す文字列のリスト
- Ipython.display
 - Javascript
 - HTML
 - display
- automation

### functions
- shutil.rmtree(path) #ディレクトリツリー全体を削除する。
- copy.deepcopy()#ただの.copy()が参照先をコピーするのに対して、deepcopy()はコンテンツもコピーする。
- json.load(f) #csvモジュールのreader()のjson版。これで、pythonのオブジェクトとしてjsonデータを扱えるようになる。
- display() #引数に変数を指定することで、そのデータフレームを表示できる。from IPython.display import displayの必要性あり。
- reset_index()　#reset_indexはデータフレームのインデックスを振り直す関数。drop引数がTrueなら、古いインデックスをカラムに追加しないように指定できる。

### matplotlib関連
- fig.canvas.draw()

- (_df["HOT"] + 0)左のように書くと、どういう処理になるのだろう。
- no_text_num = len(np.where(input_data['text'].isnull().values)[0]) #isnull().valuesでndarrayを返す。[https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.values.html]。 また、where(引数)でTrueとなるインデックスをndarrayに格納する。その数をlenで数え、格納する。

### 略語
NaN：Not a Number。浮動小数点演算の結果として、不正なオペランドを与えられた際に生じる不定の値。
inf：無限
