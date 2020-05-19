import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


house = pd.read_csv("train.csv")

house['yeni_mi'] = np.where(house['YearBuilt']>=2005, 1, 0)

Y = house["SalePrice"]
X = house[["BedroomAbvGr","yeni_mi","FullBath","GarageCars","WoodDeckSF","OverallQual","LotArea"]]


##################################################################################################################
# OLS

# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# print(results.summary())


##################################################################################################################
# Lasso
# X_egitim, X_test, y_egitim, y_test = train_test_split(X, Y)
#
# lassoregr = Lasso(alpha=10**20.5)
# lassoregr.fit(X_egitim, y_egitim)
#
# y_egitim_tahmini = lassoregr.predict(X_egitim)
# y_test_tahmini = lassoregr.predict(X_test)
# print()
# print("Eğitim kümesi R-Kare değeri       : {}".format(lassoregr.score(X_egitim, y_egitim)))
# print("-----Test kümesi istatistikleri---\n")
# print("Test kümesi R-Kare değeri         : {}".format(lassoregr.score(X_test, y_test)))
# print("Ortalama Mutlak Hata (MAE)        : {}".format(mean_absolute_error(y_test, y_test_tahmini)))
# print("Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_test_tahmini)))
# print("Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_test_tahmini)))
# print("Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_test_tahmini) / y_test)) * 100))

# eğitim kümesi R-square değeri 0 ve test kümesi R-square değeri -0.00022


##################################################################################################################
# Ridge

# X_egitim, X_test, y_egitim, y_test = train_test_split(X, Y)
#
# ridgeregr = Ridge(alpha=10**37)
# ridgeregr.fit(X_egitim, y_egitim)
#
# y_egitim_tahmini = ridgeregr.predict(X_egitim)
# y_test_tahmini = ridgeregr.predict(X_test)
#
# print("Eğitim kümesi R-Kare değeri       : {}".format(ridgeregr.score(X_egitim, y_egitim)))
# print("-----Test kümesi istatistikleri---")
# print("Test kümesi R-Kare değeri         : {}".format(ridgeregr.score(X_test, y_test)))
# print("Ortalama Mutlak Hata (MAE)        : {}".format(mean_absolute_error(y_test, y_test_tahmini)))
# print("Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_test_tahmini)))
# print("Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_test_tahmini)))
# print("Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_test_tahmini) / y_test)) * 100))


# eğitim kümesi R-Square değeri 0 ve test kümesi R-Square değeri -0.00025

##################################################################################################################
# ElasticNet

# X_egitim, X_test, y_egitim, y_test = train_test_split(X, Y)
#
# elasticregr = ElasticNet(alpha=10**21, l1_ratio=0.5)
# elasticregr.fit(X_egitim, y_egitim)
#
# y_egitim_tahmini = elasticregr.predict(X_egitim)
# y_test_tahmini = elasticregr.predict(X_test)
#
# print("Eğitim kümesi R-Kare değeri       : {}".format(elasticregr.score(X_egitim, y_egitim)))
# print("-----Test kümesi istatistikleri---")
# print("Test kümesi R-Kare değeri         : {}".format(elasticregr.score(X_test, y_test)))
# print("Ortalama Mutlak Hata (MAE)        : {}".format(mean_absolute_error(y_test, y_test_tahmini)))
# print("Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_test_tahmini)))
# print("Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_test_tahmini)))
# print("Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_test_tahmini) / y_test)) * 100))


# Eğitim kümesi R-Square değeri 0 ve test kümesi R-Square değeri -3.12 . Modelimiz overfit etti.






