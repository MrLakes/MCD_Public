# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:02:46 2021
@author: Act. Daniel Lagunas Barba
"""
# python -m pip install researchpy // en anaconda Prompt
# python -m pip install pingouin // en anaconda Prompt

# pip install researchpy // directo en la consola de python

#%% Librerias
import pandas as pd
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import pingouin as pg

#%% Datos de crecimiento de plantas.
df = pd.read_excel('C:/Users/dlagu/Python Scripts/AEM/ANOVA/Diet.xlsx')

# Traducimos y cambiamos nuestras variables.
df.columns = ['Genero', 'Dieta', 'Peso', 'Peso6Semanas']

# Agregamos la variación de los pesos para medir la eficacia de las dietas.
df['VarPeso'] = df['Peso6Semanas']-df['Peso']

# Tomamos muestras de tamaño 24 por dieta para balancear los datos
data = pd.concat([df[df['Dieta']=='A'].sample(24),
                  df[df['Dieta']=='B'].sample(24),
                  df[df['Dieta']=='C'].sample(24)])
#%%
groups = data.groupby('Dieta').count().reset_index()
groups.plot(kind = 'bar', x = 'Dieta', y = 'VarPeso')
print(rp.summary_cont(data['VarPeso'].groupby(data['Dieta'])))
data.boxplot('VarPeso', by = 'Dieta', rot = 90)

#%% One way ANOVA
#Ho: Todas las medias son iguales (p>0.05)
#Ho: Al menos 1 media es distinta (p<0.05)
model = ols('VarPeso ~ Dieta', data = data).fit()
anova_table = sm.stats.anova_lm(model, typ = 2)
print(anova_table)
print(f'Al ser{anova_table.iloc[0, 3]: .4f} menor a 0.05, entonces al menos una media es distinta.')

#%% Supuestos del modelo
#Normalidad prueba de Shapiro-Wilk
#Ho: Normalidad(p>0.05)
#H1: No normalidad (p<0.05)
#Normalidad en las variables
print(pg.normality(data, dv = 'VarPeso', group = 'Dieta'), end='\n\n')

#Homocedasticidad prueba de Levene (sin normalidad)
#Ho: Homocedasticidad (p>0.05)
#H1: No Homocedasticidad (p<0.05)
print(pg.homoscedasticity(data, dv = 'VarPeso', group = 'Dieta',
                          method = 'levene'), end='\n\n')

#Homocedasticidad prueba de Bartlett (con normalidad)
#Ho: Homocedasticidad (p>0.05)
#H1: No Homocedasticidad (p<0.05)
print(pg.homoscedasticity(data, dv = 'VarPeso', group = 'Dieta',
                          method = 'bartlett'))

#%%Comparación múltiple Prueba de Tukey
comp = mc.MultiComparison(data['VarPeso'], data['Dieta'])
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())



#%% Two ways ANOVA
#Ho: Todas las medias son iguales (p>0.05)
#Ho: Al menos 1 media es distinta (p<0.05)
anova_2 = ols('VarPeso ~ Dieta + Genero', data = data).fit()
tabla_anova_2=sm.stats.anova_lm(anova_2, typ=2)
print(tabla_anova_2)

anova_3 = ols('VarPeso ~ Dieta + Genero + Dieta:Genero', data=data).fit()
tabla_anova_3=sm.stats.anova_lm(anova_3, typ=2)
print(tabla_anova_3)

#%% Prueba de Tukey (HSD)
interaction_groups = "Dieta" + data.Dieta.astype(str) + " & " +\
                     "Genero" + data.Genero.astype(str)
comp = mc.MultiComparison(data["VarPeso"], interaction_groups)
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())
