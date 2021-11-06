# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:12:11 2021
@author: Act. Daniel Lagunas Barba
"""

# Importamos librerías a utilizar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import missingno as msno
import scipy.stats as stats

#%% Importamos nuestros datos.
df = pd.read_csv('C:/Users/dlagu/Python Scripts/AEM/1.04. Real-life example.csv')
df.info()
print(df.columns)

#%% Traducimos y cambiamos nuestras variables.
df.columns = ['Marca','Precio','Carrocería','Millaje','Motor','Combustible',
              'Registrado','Año','Modelo']

# Revisamos si hay datos faltantes.
plt.figure()
msno.bar(df)
plt.show()

#%% Eliminaremos los registros que no traen precio ni motor para trabajar con
# una base completa.
df = df.dropna(axis = 0)
plt.figure()
msno.bar(df)
plt.show()

#%% Vemos correlaciones y la distribución de los datos entre las variables.
# Gráfica de dispersión para visualizar la correlación
plt.figure()
sns.pairplot(df)
plt.show()

# Mapa de calor para representar la matriz de correlaciones
plt.figure()
sns.heatmap(df.corr(), annot = True, cmap='coolwarm')
plt.title("Matriz de correlaciones")
plt.show()

#%% Vemos de nuestras variables categóricas cuales podríamos utilizar como variables dummies
print(df['Marca'].unique())
print(df['Carrocería'].unique())
print(df['Combustible'].unique())
print(df['Registrado'].unique())

#%% Convertimos nuestras variables a dummies.
D_Marca = pd.get_dummies(df['Marca'], columns=['Marca'])
D_Carrocería = pd.get_dummies(df['Carrocería'], columns=['Carrocería'])
D_Combustible = pd.get_dummies(df['Combustible'], columns=['Combustible'])
D_Registrado = pd.get_dummies(df['Registrado'], columns=['Registrado'])

# Para el documento escrito
DMa_s = D_Marca.sum()
DCa_s = D_Carrocería.sum()
DCo_s = D_Combustible.sum()
DRe_s = D_Registrado.sum()

# Resumen de las variables dummies.
print('Resumen por marca:\n', DMa_s, '\n')
print('Resumen por carrocería:\n', DCa_s, '\n')
print('Resumen por combustible:\n', DCo_s, '\n')
print('Resumen por registro:\n', DRe_s)

#%% Modificamos nuestra base original con las variables dummies.
df2 = pd.concat([df[['Precio', 'Millaje', 'Motor', 'Año']], D_Marca.iloc[:, :-1],
                 D_Carrocería.iloc[:, :-1], D_Combustible.iloc[:, :-1],
                 D_Registrado.iloc[:, 1]], axis = 1)

'''
Dejamos afuera a:
    Marca:       Volkswagen
    Carrocería:  van
    Combustible: Petrol
    Registrado:  no   
'''

df2.rename(columns={'yes':'Registrado'}, inplace=True)
print(df2.head())

#%% Selección de variables para la regresión lineal
# Tomaremos como variable independiente el millaje del auto.
X = df2[['Millaje']]

# Variable Dependiente
Y = df2['Precio']

#%% Graficación
plt.figure()
plt.scatter(X, Y)
plt.title('Precio de autos usados por millaje')
plt.xlabel('Millaje')
plt.ylabel('Precio')
plt.show()

#%% Modelo de regresión lineal
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
rls = linear_model.LinearRegression()
rls.fit(X_train, Y_train)
Y_pred = rls.predict(X_test)
Y_pred2 = rls.predict(X_train)

# Datos de la regresión Lineal
print("Coeficiente: ", rls.coef_)
print("Intercepto: ", rls.intercept_)
print("R^2: ", rls.score(X_train, Y_train))

#%% Gráfica del modelo lineal
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred, color='r',linewidth=3)
plt.title('Regresión Lineal ')
plt.xlabel('Millaje')
plt.ylabel('Precio')
plt.show()

#%% Ajuste de la linea de regresión
plt.figure()
sns.regplot(Y_test, Y_pred, data=df2, marker='+')
plt.xlabel('Actual Values')
plt.ylabel('Predicted  Values')
plt.title('Actual Values VS Predicted Value')
plt.show()

#%% Modelo de regresión lineal (statsmodel)
# Omnibus y Jarque-Bera miden si los errores se distribuyen de manera normal.
# Durbin-Watson mide la autocorrelación de los errores.
X2 = sm.add_constant(X_train, prepend = True)
rls2 = sm.OLS(endog = Y_train, exog = X2)
rls2 = rls2.fit()
print(rls2.summary())
Y_pred3 = rls2.predict(X2)
error2 = Y_train - Y_pred3

#%% Visualizar Homocedasticidad - (La varianza de los errores es constante.)
plt.figure()
sns.regplot(Y_pred3,error2, data=df2, marker='*')
plt.xlabel('Fitted Values', size=20)
plt.ylabel('Residuals', size=20)
plt.title('Fitted Values VS Residuals', size=20)
plt.show()

#%% Forma Estadística de Homocedasticidad: (Hay igualdad entre las varianzas.)
# Breusch-Pagan
# Ho: Homocedasticidad (p>0.05)
# H1: No hay homocedasticidad (p<0.05)
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(error2, X2)
print(lzip(names, test))

#%% Forma gráfica de la  normalidad de los residuos
plt.figure()
plt.title('Histograma de los residuos')
plt.hist(rls2.resid_pearson)
plt.show()

#%% QQ plot
plt.figure()
plt.title('Gráfico Q-Q')
ax = sm.qqplot(rls2.resid_pearson, line = "45")
plt.show()

#%% Forma Estadística de la normalidad (Shaphiro-Wilk)
# Shapiro-Wilks también mide la normalidad de los errores.
names = ['Stastistic', 'p-value']
test = stats.shapiro(error2)
print(lzip(names, test))




#%%Regresión Lineal Múltiple
#Variables independientes
X_multiple = pd.concat([df2.iloc[:, 1], df2.iloc[:, 3:]], axis = 1)
#Variable dependiente
Y_multiple = df2['Precio']

#Crear el modelo
X_train, X_test, Y_train, Y_test = train_test_split(X_multiple, Y_multiple, test_size = 0.2)
X = sm.add_constant(X_train, prepend = True)
rlm = sm.OLS(endog = Y_train, exog = X,)
rlm = rlm.fit()
print(rlm.summary())

#%%Calcular el error
Y_pred = rlm.predict(X)
error = Y_train-Y_pred

#%% Ajuste de la linea de regresión
plt.figure()
sns.regplot(Y_train, Y_pred, data=df2, marker='+')
plt.xlabel('Actual Values')
plt.ylabel('Predicted  Values')
plt.title('Actual Values VS Predicted Value')
plt.show()

#%% Forma Estadística de Homocedasticidad
#Breusch-Pagan
#H0: Homocedasticidad (p>0.05)
#H1: No homocedasticidad (p<0.05)
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(error, X)
print(lzip(names, test))

#%% Forma estadística de la normalidad (Shapiro-Wilk)
#Ho: Normalidad (p>0.05)
#H1: No normalidad (p<0.05)
names = [' Statistic', 'p-value']
test = stats.shapiro(error)
print(lzip(names,test))
