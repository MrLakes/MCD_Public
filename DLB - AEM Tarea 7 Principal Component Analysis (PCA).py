# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:33:11 2021
@author: Act. Daniel Lagunas Barba
Principal Component Analysis (PCA)
"""
# Librerías
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#%% Base de datos de Iris.
iris = load_iris()
print(iris['DESCR'])
data = pd.DataFrame(iris.data)
data.columns = iris.feature_names

# Correlación entre las variables
cor = data.corr()
# Mapa de calor para representar la matriz de correlaciones
plt.figure()
sns.heatmap(cor, annot = True, cmap='coolwarm')
plt.title("Matriz de correlaciones")
plt.show()

# Forma gráfica (pocas variables)
plt.figure()
sns.pairplot(data)
plt.show()
print()

# Media de las variables
print("Medias:\n", data.mean())
print()

# Varianza de las variables
print("Varianzas:\n", data.var())

#%% Opción 1: Estandarizar los datos media 0 y desviación estándar 1
# Normales y sin outliers
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)

#%% Algoritmo PCA
pca = PCA()
# pca = PCA(n_components=10) # Indicar el número de componentes principales 
pca.fit(scaled_data)

# Ponderación de los componentes principales (vectores propios)
pca_score = pd.DataFrame(data = pca.components_, columns = data.columns,)

# Mapa de calor para visualizar in influencia de las variables
plt.figure()
sns.heatmap(pca_score.transpose(), annot = True, cmap='plasma')
plt.title("Influencia de variables por componente")
plt.grid(False)
plt.show()

#%% Gráfica del aporte a cada componente principal
# Aporte al primer componente principal 
matrix_transform = pca.components_.T
plt.bar(np.arange(4), matrix_transform[:,0])
plt.xlabel('Num variable real')
plt.ylabel('1° Vector asociado')
plt.show()

plt.bar(np.arange(4), matrix_transform[:,1])
plt.xlabel('Num variable real')
plt.ylabel('2° Vector asociado')
plt.show()

plt.bar(np.arange(4), matrix_transform[:,2])
plt.xlabel('Num variable real')
plt.ylabel('3° Vector asociado')
plt.show()

plt.bar(np.arange(4), matrix_transform[:,3])
plt.xlabel('Num variable real')
plt.ylabel('4° Vector asociado')
plt.show()

#%% Nuevas variables, components principales
pca_data = pca.transform(scaled_data) 

# Porcentaje de varianza explicada por cada componente principal proporciona
# Lambda/suma_Lambda (valor_propio/suma_valores_propios)
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
print(per_var)

# Porcentaje de varianza acumulado de los componentes
porcent_acum = np.cumsum(per_var)
print(porcent_acum)

#%% Graficar componentes principales     
mglearn.discrete_scatter(pca_data[:,0], pca_data[:,1], iris.target)
plt.legend(iris.target_names,loc = 'best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

