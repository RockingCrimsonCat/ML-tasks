import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Генерація даних
np.random.seed(0)
num_applicants = 1600

math_scores = np.random.normal(loc=150, scale=30, size=num_applicants).clip(0, 200)
english_scores = np.random.normal(loc=150, scale=30, size=num_applicants).clip(0, 200)
ukrainian_scores = np.random.normal(loc=150, scale=30, size=num_applicants).clip(0, 200)

privileges = np.random.choice([0, 1], size=num_applicants, p=[0.9, 0.1])

data = pd.DataFrame({
    'Math Score': math_scores,
    'English Score': english_scores,
    'Ukrainian Score': ukrainian_scores,
    'Privileged': privileges
})

data.to_csv('data.csv', index=False, sep=',')
data.head()

# Підготовка даних
data = pd.read_csv('applicants_data.csv', sep=',')

data['Rating'] = 0.4 * data['Math Score'] + 0.3 * data['English Score'] + 0.3 * data['Ukrainian Score']
data['Admitted'] = False

data.loc[(data['Privileged'] == 0) & (data['Rating'] >= 160) & (data['Math Score'] >= 140), 'Admitted'] = True
data.loc[(data['Privileged'] == 1) & (data['Rating'] >= 144) & (data['Math Score'] >= 126) & (data['English Score'] >= 126) & (data['Ukrainian Score'] >= 126), 'Admitted'] = True

sorted_data = data.sort_values(by='Rating', ascending=False)

non_privileged_quota = 315
privileged_quota = 35

admitted_candidates = pd.concat([
    sorted_data[sorted_data['Privileged'] == 0].head(non_privileged_quota),
    sorted_data[sorted_data['Privileged'] == 1].head(privileged_quota)
])

data['Admitted'] = False
data.loc[admitted_candidates.index, 'Admitted'] = True

data.to_csv('complete_candidates_list.csv', index=False, sep=',')

data.head()

# Нормалізація даних
X = data[['Math Score', 'English Score', 'Ukrainian Score']]
y = data['Admitted'].astype(int).values

ss = StandardScaler()
X = ss.fit_transform(X)

# Поділ даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape

# Функції
def create_model(layers, neurons, optimizer, activation):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))  # Corrected input shape
    for i in range(layers):
        model.add(Dense(neurons, activation=activation))  # Removed redundant +1
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_models(models_parameters, X_train, y_train, X_test, y_test):
    results = {}
    detailed_results = []
    for i, params in enumerate(models_parameters, 1):
        layers, neurons, optimizer_instance, activation = params
        model = create_model(layers, neurons, optimizer_instance, activation)
        print(f"Training Model {i}...")
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        results[f'Model {i}'] = evaluation[1]
        detailed_results.append({
            'Model': f'Model{i}',
            'Layers': layers,
            'Neurons per Layer': neurons,
            'Optimizer': optimizer_instance.__class__.__name__,
            'Activation': activation,
            'Accuracy': evaluation[1],
            'Loss': evaluation[0]
        })

    for result in detailed_results:
        print(f"{result['Model']} - Layers: {result['Layers']}, Neurons: {result['Neurons per Layer']}, "
              f"Optimizer: {result['Optimizer']}, Activation: {result['Activation']}, "
              f"Accuracy: {result['Accuracy']*100:.2f}%, Loss: {result['Loss']:.2f}")

    best_model = max(results, key=lambda x: results[x])
    best_accuracy = results[best_model]
    best_params = models_parameters[int(best_model.split(' ')[1]) - 1]

    print(f"\nBest model {best_model} with an accuracy of {best_accuracy*100:.2f}%.")
    print(f"Parameters: Layers: {best_params[0]}, Neurons per Layer: {best_params[1]}, Optimizer: {best_params[2].__class__.__name__}, "
          f"Activation: {best_params[3]}")

# Порівняння кількості нейронів
models_parameters = [
    (2, 16, Adam(), 'relu'),
    (2, 32, Adam(), 'relu'),
    (2, 64, Adam(), 'relu'),
    (2, 128, Adam(), 'relu'),
    (2, 256, Adam(), 'relu'),
    (2, 512, Adam(), 'relu'),
]
train_and_evaluate_models(models_parameters, X_train, y_train, X_test, y_test)

# Порівняння шарів
models_parameters = [
    (3, 16, Adam(), 'relu'),
    (4, 16, Adam(), 'relu'),
    (5, 16, Adam(), 'relu'),
    (6, 16, Adam(), 'relu'),
    (7, 16, Adam(), 'relu'),
    (8, 16, Adam(), 'relu'),
    (9, 16, Adam(), 'relu'),
    (10, 16, Adam(), 'relu'),
]
train_and_evaluate_models(models_parameters, X_train, y_train, X_test, y_test)

# Пошук найкращої кількості епох
best = create_model(2, 16, Adam(), 'relu')
for i in range(50, 500, 50):
    best.fit(X_train, y_train, epochs=i, batch_size=10, verbose=0)
    print(i, best.evaluate(X_test, y_test, verbose=0))

# Фінальна модель
best = create_model(2, 16, Adam(), 'relu')
history = best.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0, validation_split=0.4)
best.evaluate(X_test, y_test, verbose=0)

# Структура моделі
best.summary()
#Графіки
plt.figure(figsize=(14,7))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
