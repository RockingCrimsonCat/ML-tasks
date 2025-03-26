import pandas as pd
nyc = pd.read_csv(r'D:\Labs ML\data_NY.csv')
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)
print(nyc.head(3))
#Розбиття даних для навчання і тестування
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11)
print('Перевірка пропорції навчальних тестових даних:')
print(x_train.shape)
print(x_test.shape)
train_size = x_train.shape[0]
test_size = x_test.shape[0]
total_size = train_size + test_size
train_percentage = (train_size / total_size) * 100
test_percentage = (test_size / total_size) * 100
print('Відсоток навчальних даних:', train_percentage)
print('Відсоток тестових даних:', test_percentage)
from sklearn.linear_model import LinearRegression
#Навчання моделі
linear_regression = LinearRegression()
linear_regression.fit(X=x_train, y=y_train)
print(linear_regression.fit(X=x_train, y=y_train))
print('Кут нахилу:', linear_regression.coef_)
print('Точка перетину:', linear_regression.intercept_)
#Тестування моделі
predicted = linear_regression.predict(x_test)
expected = y_test
for p,e in zip(predicted[::5], expected[::5]):
    print(f'cпрогнозоване: {p:.2f}, очікуване: {e:.2f}')
#Прогнозування майбутніх температур і оцінка минулих температур
predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)
print(predict(2022))
print(predict(1893))
#Візуалізація набору даних з регресійною прямою
import seaborn as sns
import matplotlib.pyplot as plt
axes = sns.scatterplot(data=nyc, x='Date', y='Temperature', hue='Temperature', palette='winter', legend=False)
axes.set_ylim(10, 70)
import numpy as np
x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
y = predict(x)
line = plt.plot(x,y)
plt.show()

#2 згенеруйте набір даних та класифікуйте його використавши класифікатор SVC.

from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] >0 )
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.show()
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                  y=X[y == cl, 1],
                  alpha=0.8, c=colors[idx],
                  marker=markers[idx],
                  label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, label='test set')

svn = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svn.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svn)
plt.legend(loc='upper left')
plt.show()

