import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

y = y.reshape((len(y), 1))
# feature scaling
from sklearn.preprocessing import StandardScaler

ss1 = StandardScaler()  # сколько среднеквадратичных отклонений содержит наша величина
X = ss1.fit_transform(X)  # применяем к тестовой выборке
ss2 = StandardScaler()  # сколько среднеквадратичных отклонений содержит наша величина
y = ss2.fit_transform(y)  # применяем к тестовой выборке

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#predict and inverse transform
result=regressor.predict(ss1.transform([[6.5]]))
print(ss2.inverse_transform(result))


plt.scatter(ss1.inverse_transform(X),ss2.inverse_transform(y),color='red')
plt.plot(ss1.inverse_transform(X),ss2.inverse_transform(regressor.predict(X)),color='blue')
plt.title('Truth or Bluff (support vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


X_grid=np.arange(min(X),max(X)+1,0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(ss1.inverse_transform(X),ss2.inverse_transform(y),color='red')
plt.plot(ss1.inverse_transform(X_grid),ss2.inverse_transform(regressor.predict(X_grid)),color='blue')
plt.title('Truth or Bluff (support vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()