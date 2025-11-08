# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 21:59:20 2025

@author: gabri
"""
#1
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
path = "C:/Users/gabri/Desktop/EmpleadosAttritionFinal.csv" #ruta de exportar

# %%
#2

EmpleadosAttrition = pd.read_csv(r"C:\Users\gabri\Documents\Data Science\15. Ingeniería de Características\empleadosRETO.csv")
# %%
#3

EmpleadosAttrition.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], inplace=True)
# %%
#4-5

EmpleadosAttrition['HiringDate'] = EmpleadosAttrition['HiringDate'].replace('2/30/2012', '2/29/2012') #no existe 30 de febrero
EmpleadosAttrition['Year'] = pd.to_datetime(EmpleadosAttrition['HiringDate']).dt.year.astype(int)
# %%
#6

EmpleadosAttrition['YearsAtCompany'] = 2018 - EmpleadosAttrition['Year']
# %%
#7

EmpleadosAttrition['DistanceFromHome'] = EmpleadosAttrition['DistanceFromHome'].astype(str).str.replace(' km', '')
EmpleadosAttrition['DistanceFromHome'] = EmpleadosAttrition['DistanceFromHome'].astype(int)
# %%
#8

EmpleadosAttrition.rename(columns={'DistanceFromHome':'DistanceFromHome_km'}, inplace=True) #8
# %%
#9

EmpleadosAttrition['DistanceFromHome'] = EmpleadosAttrition['DistanceFromHome_km'] 
# %%
#10

EmpleadosAttrition.drop(columns=['Year', 'DistanceFromHome_km', 'HiringDate'], inplace=True)
# %%
#11

SueldoPromedioDepto = pd.DataFrame(EmpleadosAttrition.groupby(['Department'])['MonthlyIncome'].mean())
# %%
#12

escalador = preprocessing.MinMaxScaler()
EmpleadosAttrition['MonthlyIncome'] = escalador.fit_transform(EmpleadosAttrition[['MonthlyIncome']]) #12
# %%
#13

# BusinessTravel
BT_OH = pd.get_dummies(EmpleadosAttrition.BusinessTravel, prefix='BusinessTravel', dtype=int)
EmpleadosAttrition = pd.concat([EmpleadosAttrition, BT_OH], axis=1)

# Department
DEP_OH = pd.get_dummies(EmpleadosAttrition.Department, prefix='Department', dtype=int)
EmpleadosAttrition = pd.concat([EmpleadosAttrition, DEP_OH], axis=1)

# EducationField
EDU_OH = pd.get_dummies(EmpleadosAttrition.EducationField, prefix='EducationField', dtype=int)
EmpleadosAttrition = pd.concat([EmpleadosAttrition, EDU_OH], axis=1)

# Gender
GEN_OH = pd.get_dummies(EmpleadosAttrition.Gender, prefix='Gender', dtype=int)
EmpleadosAttrition = pd.concat([EmpleadosAttrition, GEN_OH], axis=1)

# JobRole
JR_OH = pd.get_dummies(EmpleadosAttrition.JobRole, prefix='JobRole', dtype=int)
EmpleadosAttrition = pd.concat([EmpleadosAttrition, JR_OH], axis=1)

# MaritalStatus
MS_OH = pd.get_dummies(EmpleadosAttrition.MaritalStatus, prefix='MaritalStatus', dtype=int)
EmpleadosAttrition = pd.concat([EmpleadosAttrition, MS_OH], axis=1)

EmpleadosAttrition['Attrition'] = EmpleadosAttrition['Attrition'].astype(str)
EmpleadosAttrition['Attrition'] = EmpleadosAttrition['Attrition'].replace({'Yes': 1, 'No': 0}).astype(float)

EmpleadosAttrition['OverTime'] = EmpleadosAttrition['OverTime'].astype(str)
EmpleadosAttrition['OverTime'] = EmpleadosAttrition['OverTime'].replace({'Yes': 1, 'No': 0}).astype(float)

# Borrar las columnas categóricas originales
EmpleadosAttrition.drop(columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus'], inplace=True)
# %% 
#14

correlacion = EmpleadosAttrition.corr()['Attrition']
correlacion = abs(correlacion)
columnas = correlacion[correlacion >= 0.1].index
# %%
#15

EmpleadosAttritionFinal = EmpleadosAttrition[columnas].copy()

# %%
#16

pca = PCA()
EmpleadosAttritionPCA = pca.fit_transform(EmpleadosAttritionFinal)
print(pca.explained_variance_ratio_)

# %%
#17

#insert para que C0 y C1 queden al inicio
C0 = EmpleadosAttritionPCA[:,0]
C1 = EmpleadosAttritionPCA[:,1]
EmpleadosAttritionFinal.insert(0, 'C0', C0)
EmpleadosAttritionFinal.insert(1, 'C1', C1)


# %%
#18
#revisión de los df en consola y exportar como csv

EmpleadosAttritionFinal.to_csv(path, index=False)

#print(SueldoPromedioDepto)
#print(EmpleadosAttrition.columns)
#print(EmpleadosAttrition)
#print(EmpleadosAttritionFinal)
