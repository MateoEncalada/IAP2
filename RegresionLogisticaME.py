# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:30:49 2024

@author: maten
"""

import numpy as np  # Trabajar con datos y operaciones matemáticas
import matplotlib.pyplot as plt  # Graficar visualizaciones
import pandas as pd  # Manejo de datos en tablas (DataFrames)



# Cargar el dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

X =dataset.iloc[:,[2,3]]##todas las filas pero las columnas dos y tres

y =dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split  # División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler ##escalables para que sean comparables
# Escalado de variables
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##Aprende a predecir las clasificaciones
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

##Ahora vamos a validar que el modelo que esta en prediccine este bien
from sklearn.metrics import confusion_matrix ##nps permite ver que tan bueno es el modelo
conmet = confusion_matrix(y_test, y_pred)##65 no compraron 24 van comprar| 8 y 3 posibles errores del modelo es decir si sumamos el 11% de probabilidad de error 

from matplotlib.colors import ListedColormap

# Visualizar el algoritmo de entrenamiento gráficamente
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title("Logistic Regression (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# Visualizar el algoritmo de prueba gráficamente
X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title("Logistic Regression (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

