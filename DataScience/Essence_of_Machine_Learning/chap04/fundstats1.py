import numpy as np

# 入手元：気象庁ホームページ
# （https://www.data.jma.go.jp/gmd/risk/obsdl/index.php）
# 2018年4月の東京の最高気温（日別）
x = np.array([21.9, 24.5, 23.4, 26.2, 15.3, 22.4, 21.8, 16.8,
              19.9, 19.1, 21.9, 25.9, 20.9, 18.8, 22.1, 20.0,
              15.0, 16.0, 22.2, 26.4, 26.0, 28.3, 18.7, 21.3,
              22.5, 25.0, 22.0, 26.1, 25.6, 25.7])

m = x.sum() / len(x)
s = np.sqrt(((x - m)**2).sum() / len(x))
print("平均値:{:.4f}".format(m))
print("標準偏差:{:.4f}".format(s))
