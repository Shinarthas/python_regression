import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

# зашивруем категории в виде вектора
# закодируем зависимую переменную
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 1 элемент - вид трансвормации, 2 - класс странсформера, 3 - колонка для кодирования
# remainder - инструкция для трансформации, говорит расширить

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
# разодьем данные на тестовые и проверочные
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)  # random_state=1 убирает рандом что он всегжа одинаков

# training simple regression model
from sklearn.linear_model import LinearRegression

lg = LinearRegression()
lg.fit(X_train, y_train)



# predict the results
predicted_y=lg.predict(X_test)
print(len(y_test),len(predicted_y))
np.set_printoptions(precision=2)
print(np.concatenate(
    (y_test.reshape(len(y_test), 1),
     predicted_y.reshape(len(predicted_y), 1),
     abs(y_test-predicted_y).reshape(len(y_test), 1)
     ),
    1))
# visualise training results
plt.scatter(range(len(y_test)), y_test, color='red')  # просто ставим точки
plt.scatter(range(len(predicted_y)), predicted_y,
            color='blue')  # просто рисуем линию но она будет прямая, так как мы исп метод predict  на тестовой
plt.title('Real results vs Model')
plt.xlabel('N')
plt.ylabel('profit')
plt.show()

#Getting the final linear regression equation with the values of the coefficients
print(lg.coef_)
print(lg.intercept_)

print(lg.predict([[1.0,0.0,0.0,160000,130000,300000]]))