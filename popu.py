import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

names=['date','population','ROC']

df1=pd.read_csv("population.csv",header=None)
df1.columns=names
df1.drop(df1.tail(75).index,inplace=True)
del df1['ROC']


x=df1.iloc[:,:-1].values
y=df1.iloc[:,1].values

date1=np.asarray(df1.date)
population1=np.asarray(df1.population)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=0)


print(x_train,x_test,y_train,y_test)

reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
a=reg.coef_
b=reg.intercept_

y_pred=reg.predict(x_test)

data=pd.DataFrame(y_test,y_pred)
print(data)








