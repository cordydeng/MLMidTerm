import numpy as np
import pandas as pd
import seaborn as sb

dataset=pd.read_csv("train.csv")
sb.jointplot(data=dataset,x="CentralAir",y="SalePrice")
sb.jointplot(data=dataset,x="PoolArea",y="SalePrice")#由圖可知PoolArea和房價沒有太大關係
dataset.drop(["Street","Alley","Utilities","PoolArea","PoolQC","Fence","MiscFeature","MiscVal"],axis=1,inplace=True)
#Street幾乎都是一樣Alley超過90%NA Utilities都一樣 Fence和MiscFeature缺值過多
dataset.head()
dataset.info()
dataset["LotFrontage"].mean()
dataset["LotFrontage"]=dataset["LotFrontage"].fillna(dataset["LotFrontage"].mean())

dataset["MasVnrType"].value_counts().idxmax()
dataset["MasVnrType"].fillna(dataset["MasVnrType"].value_counts().idxmax(),inplace=True)

dataset["MasVnrArea"].value_counts().idxmax()
dataset["MasVnrArea"].fillna(dataset["MasVnrArea"].value_counts().idxmax(),inplace=True)

dataset["BsmtQual"].fillna("No",inplace=True)
dataset["BsmtCond"].fillna("No",inplace=True)
dataset["BsmtExposure"].fillna("Nb",inplace=True)
dataset["BsmtFinType1"].fillna("Nb",inplace=True)
dataset["BsmtFinType2"].fillna("Nb",inplace=True)

dataset["Electrical"].value_counts().idxmax()
dataset["Electrical"].fillna(dataset["Electrical"].value_counts().idxmax(),inplace=True)

dataset["FireplaceQu"].fillna("Nf",inplace=True)
dataset["GarageType"].fillna("Ng",inplace=True)
dataset["GarageYrBlt"].fillna(0,inplace=True)
dataset["GarageFinish"].fillna("Ng",inplace=True)
dataset["GarageQual"].fillna("Ng",inplace=True)
dataset["GarageCond"].fillna("Ng",inplace=True)

miss=dataset.isnull().sum()!=1460
print(dataset.isnull().sum().to_string())

ds=pd.get_dummies(data=dataset)
print(ds.isnull().sum().to_string())
ds.head()
ds.info()
ds.corr()
x=ds.drop(["SalePrice"],axis=1)
y=ds["SalePrice"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=54)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
prediction=lr.predict(x_test)
prediction

import joblib
joblib.dump(lr,"AdvanceHousePricing.pkl",compress=3)