# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:20:22 2024

@author: maten
SVM Super Vector Machine
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

##
from sklearn.svm import SVC

svma = SVC(kernel = "linear", random_state=0)

svma.fit(X_train, y_train)

y_pred = svma.predict(X_test)
