import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

# training simple regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
# в какую степень возводить
pf=PolynomialFeatures(degree=4)
X_poly=pf.fit_transform(X)
lin_reg_2=LinearRegression()
result_poly=lin_reg_2.fit(X_poly,y)


plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(X_poly),color='blue')
plt.plot(X,lin_reg.predict(X),color='green')

plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predict single
print(lin_reg_2.predict(pf.fit_transform([[2]])))