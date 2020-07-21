import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

# разодьем данные на тестовые и проверочные
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)  # random_state=1 убирает рандом что он всегжа одинаков

# training simple regression model
from sklearn.linear_model import LinearRegression

lg = LinearRegression()
lg.fit(X_train, y_train)

result = lg.predict(X_test)
print(result, y_test)

# visualise training results
plt.scatter(X_train,y_train,color='red')# просто ставим точки
plt.plot(X_train,lg.predict(X_train), color='blue')# просто рисуем линию но она будет прямая, так как мы исп метод predict  на тестовой
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years od experience')
plt.ylabel('salary')
plt.show()


# visualise test results
plt.scatter(X_test,y_test,color='red')# просто ставим точки
plt.plot(X_train,lg.predict(X_train), color='blue')# просто рисуем линию но она будет прямая, так как мы исп метод predict  на тестовой
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years od experience')
plt.ylabel('salary')
plt.show()

#Making a single prediction
print(lg.predict([[12]]))
#Getting the final linear regression equation with the values of the coefficients
print(lg.coef_)
print(lg.intercept_)

