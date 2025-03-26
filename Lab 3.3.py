import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
# Завантаження даних
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Розділення даних на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Класифікатори
classifiers = {
    "K Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "Gaussian Naive Bayes": GaussianNB()
}
# Тренування та оцінка класифікаторів
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy)
    print()
# Візуалізація
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X[:, :2], y)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    ax[i].contourf(xx, yy, Z, alpha=0.8)
    ax[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax[i].set_title(name)
colors = ['blue', 'green', 'orange']
classes = iris.target_names

for i, cls in enumerate(classes):
    ax[0].scatter([], [], c=colors[i], label=cls, edgecolors='k')

ax[0].legend()
plt.show()