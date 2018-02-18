from scipy.misc import comb
import math
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# アンサンブルの誤分類率を二項分布の確率質量関数として計算する
#-------------------------------------------------------------------------------
def ensemble_error(n_classifier, error):
    """
    アンサンブルの誤分類率を二項分布の確率質量関数として計算する
    【引数】
    n_classifier: 分類器の数
    error: 分類器の誤分類率
    """
    k_start = int(math.ceil(n_classifier / 2.0))
    # 誤分類する分類器の個数がkになる確率を計算する
    probs = [comb(n_classifier, k) * error ** k * (1 - error) ** (n_classifier - k) for k in range(k_start, n_classifier + 1)]
    # 要素をすべて足すことで、誤分類する分類器の個数が「k以上」になる確率を計算する
    return sum(probs)

#--- アンサンブルの誤分類率をグラフで図示する
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

plt.plot(error_range, ens_errors, label="Ensemble error", linewidth=2)
plt.plot(error_range, error_range, linestyle="--", label="Base error", linewidth=2)
plt.xlabel("Base error")
plt.ylabel("Base/Ensemble error")
plt.legend(loc="upper left")
plt.grid()
plt.show()
