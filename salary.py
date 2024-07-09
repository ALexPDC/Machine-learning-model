import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Crear un dataset ficticio
data = {
    'Age': [22, 35, 26, 27, 32, 45, 21, 23, 42, 57, 36, 29, 41, 39, 20],
    'EstimatedSalary': [19000, 20000, 43000, 57000, 76000, 58000, 52000, 79000, 15000, 82000, 18000, 83000, 74000, 78000, 83000],
    'Purchased': [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Mostrar los primeros registros del dataset
print(df.head())

# Visualización de la distribución de los datos
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=df)
plt.title('Distribución de Compras basadas en Edad y Salario Estimado')
plt.show()

# Características y etiquetas
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Crear el modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualizar la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
