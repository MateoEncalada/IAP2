# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:54:27 2024

@author: maten
random forest habla de la cantidad de arbiles que yo quiera ya no es solo uno

1. Selecionar un numero aleatorio de K puntos del train
2. De esos puntos se construye un arbol de desicion a esos Kpuntos de datos
3. Elegimos unnumero NTree de arboles que queremos construir y repetir los pasos 1 y 2
4. Para clasificar un nuevo punto, hacer que cada uno de los NTree=numero de arboles que quiero formar con esos puntos 
por default el algoritmo toma 10 arboles, arboles elabore
a que categoria pertenece y asignar el nuevo punto a la categoria con mas votos

Nos recomienda leer Real Time Human Recognition in parts from Single Depth
"""
import numpy as np  # Trabajar con datos y operaciones matemáticas
import matplotlib.pyplot as plt  # Graficar visualizaciones
import pandas as pd  # Manejo de datos en tablas (DataFrames)

# Cargar el dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

X =dataset.iloc[:,[2,3]].values##todas las filas pero las columnas dos y tres

y =dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split  # División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, criterion= "entropy", random_state=0)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

from sklearn.metrics import confusion_matrix ##nps permite ver que tan bueno es el modelo
conmet = confusion_matrix(y_test, y_pred)

### Visualizar el algotirmo de train graficamente con los resultados

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 500))

plt.contourf(X1, X2 , rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title("Random Forest (Training set)")

plt.xlabel("Age")

plt.ylabel("Estimated Salary")

plt.legend()

plt.show()

### Visualizar el algotirmo de test graficamente con los resultados

from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 500))

plt.contourf(X1, X2 , rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     

plt.xlim(X1.min(),X1.max())

plt.ylim(X2.min(),X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title("Random Forest (Test set)")

plt.xlabel("Age")

plt.ylabel("Estimated Salary")

plt.legend()

plt.show()
