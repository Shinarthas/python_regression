import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(random_state=0)# убираем рандом что-бы результат всегда был одинаков
dtr.fit(X,y)

X_new=np.arange(min(X),max(X),0.01)
X_new=X_new.reshape((len(X_new),1))
plt.scatter(X,y,color='red')
plt.plot(X_new,dtr.predict(X_new),color='blue')


plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(dtr.predict([[6.5]]))