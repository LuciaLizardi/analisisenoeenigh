# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:08:24 2023

@author: lucia
"""

#Manejo de Datos para análisis de inferencia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

enoe=pd.read_csv('E:/Users/lucia/ProyectoInferencia/enoe_limpia19a23.csv')


enoe['eda'] = pd.to_numeric(enoe['eda'], errors='coerce')
enoe['clase1'] = pd.to_numeric(enoe['clase1'], errors='coerce')
enoe['per'] = enoe['per'].astype(str)


#PEA > 60 AÑOS 2019
enoe_amT119=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('119'))]
enoe_amT219=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('219'))]
enoe_amT319=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('319'))]
enoe_amT419=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('419'))]
enoe_am19=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('19'))]

#PEA > 60 AÑOS 2022
enoe_amT122=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('122'))]
enoe_amT222=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('222'))]
enoe_amT322=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('322'))]
enoe_amT422=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('422'))]
enoe_am22=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (  (enoe['per'].str.contains('122') )
                                                                |(enoe['per'].str.contains('222'))
                                                                |(enoe['per'].str.contains('322') )
                                                                |(enoe['per'].str.contains('422') )
                                                                  )]
#PEA > 60 AÑOS 2023
enoe_amT123=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('123'))]
enoe_amT223=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('223'))]
enoe_am23=enoe[(enoe['eda'] >= 60) & (enoe['clase1'] == 1) & (enoe['per'].str.contains('23'))] 

#2019
print("Tamaño de muestra 1er Trimestre 2019",len(enoe_amT119))
print("Tamaño de muestra 2do Trimestre 2019",len(enoe_amT219))
print("Tamaño de muestra 3er Trimestre 2019",len(enoe_amT319))
print("Tamaño de muestra 4to Trimestre 2019",len(enoe_amT419))
print("Tamaño de muestra 2019",len(enoe_am19))

#2022
print("Tamaño de muestra 1er Trimestre 2022",len(enoe_amT122))
print("Tamaño de muestra 2do Trimestre 2022",len(enoe_amT222))
print("Tamaño de muestra 3er Trimestre 2022",len(enoe_amT322))
print("Tamaño de muestra 4to Trimestre 2022",len(enoe_amT422))
print("Tamaño de muestra 2022",len(enoe_am22))

#2023
print("Tamaño de muestra 1er Timestre 2023",len(enoe_amT123))
print("Tamaño de muestra 2do Timestre 2023",len(enoe_amT223))
print("Tamaño de muestra 2023",len(enoe_am23))


#Plots 2019
plt.hist(enoe_am19['eda'], bins=30, edgecolor='black', color='gold')
# Agrega etiquetas y título
plt.xlabel('Valores - Edad')
plt.ylabel('Frecuencia')
plt.title('Distribución de edad de PEA >= 60 Años 2019')
plt.show()

#Plots 2022
plt.hist(enoe_am22['eda'], bins=30, edgecolor='black', color='skyblue')
# Agrega etiquetas y título
plt.xlabel('Valores - Edad')
plt.ylabel('Frecuencia Absoluta')
plt.title('Distribución de edad de PEA >= 60 Años 2022')
plt.show()

#Plots 2023
plt.hist(enoe_am23['eda'], bins=30, edgecolor='black', color='salmon')
# Agrega etiquetas y título
plt.xlabel('Valores - Edad')
plt.ylabel('Frecuencia Absoluta')
plt.title('Distribución de edad de PEA >= 60 Años 2023')
plt.show()


#Nivel De Ingresos
#2019
print("Nivel de Ingresos  2019",enoe_am19['ing7c'].value_counts())
#2022
print("Nivel de Ingresos  2022",enoe_am22['ing7c'].value_counts())
#2023
print("Nivel de Ingresos  2023",enoe_am23['ing7c'].value_counts())

#Duración de la jornada laboral 
#2019
print("Duración de la jornada laboral 2019",enoe_am19['dur9c'].value_counts())
#2022
print("Duración de la jornada laboral 2022",enoe_am22['dur9c'].value_counts())
#2023
print("Duración de la jornada laboral 2023",enoe_am23['dur9c'].value_counts())


#Condición de ocupación
#2019
print("Condición de Ocupación 2019",enoe_am19['c_ocu11c'].value_counts())
#2022
print("Condición de Ocupación 2022",enoe_am22['c_ocu11c'].value_counts())
#2023
print("Condición de Ocupación 2023",enoe_am23['c_ocu11c'].value_counts())


#Prestaciones de Salud
#2019
print("Prestaciones de Salud 2019",enoe_am19['medica5c'].value_counts())
#2022
print("Prestaciones de Salud 2022",enoe_am22['medica5c'].value_counts())
#2023
print("Prestaciones de Salud 2023",enoe_am23['medica5c'].value_counts())


#Edades
#2019
print("Edades 2019",enoe_am19['eda'].value_counts())
#2022
print("Edades 2022",enoe_am22['eda'].value_counts())
#2023
print("Edades 2023",enoe_am23['eda'].value_counts())

#Media y Varianza Edades 2019
media_ed19 = 67.4
print("Media de Edades 2019",media_ed19)
var_ed19 =enoe_am19['eda'].var()
print("Varianza Edades 2019",var_ed19)


