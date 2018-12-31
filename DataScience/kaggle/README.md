データ解析の手順

1. df = pd.read_csv('hoge.csv')  
(1. profile = pandas_profiling.ProfileReport(df))
(1. profile.to_file('ProfileReport___thema.html'))
1. df.info()
1. カテゴリカル列と数値列にdfを分ける。
1. それぞれのdf.describe()

### 時系列分析
1. [季節調節分析](https://data.gunosy.io/entry/statsmodel_trend)  
モデルの概要：観測値 = トレンド成分 + 季節成分 + ノイズ成分

1週間周期の集約
import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(freq=7)

終わったら、statinarityの検定。モデルの構築。