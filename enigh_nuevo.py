# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:59:33 2023

@author: lucia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics
from scipy.stats import chi2
from scipy.optimize import curve_fit
from scipy.stats import f
from scipy.stats import norm
from matplotlib.ticker import ScalarFormatter
from scipy.stats import t


#enigh 2022
enigh_22_ing=pd.read_csv('E:/Users/lucia/ProyectoInferencia/ENIGH/ingresos.csv')
enigh_22_pob=pd.read_csv('E:/Users/lucia/ProyectoInferencia/ENIGH/poblacion.csv')

print(enigh_22_ing.columns)
print(enigh_22_pob.columns)

enigh_22 = pd.merge(enigh_22_ing, enigh_22_pob, on='folioviv', how='inner')

print(enigh_22.columns)

var_interes=['folioviv','edad','sexo','disc_ver','disc_oir','hor_1','prob_sal','numren_x','mes_1','mes_2',
             'mes_3','mes_4','mes_5','mes_6','ing_1','ing_2','ing_3','ing_4','ing_5','ing_6']

enigh_filtr=enigh_22.loc[:, var_interes]

enigh_final=enigh_filtr[(enigh_filtr['edad'] >= 60)]
# Definir un diccionario de correspondencia entre números de mes y nombres de mes
nombre_mes = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'}

# Crear columnas vacías para cada mes del 2 al 10
for num_mes, nombre in nombre_mes.items():
    enigh_final[nombre] = np.nan

print(enigh_final.columns)

enigh_final['mes_1'] = pd.to_numeric(enigh_final['mes_1'], errors='coerce')
enigh_final['mes_2'] = pd.to_numeric(enigh_final['mes_2'], errors='coerce')
enigh_final['mes_3'] = pd.to_numeric(enigh_final['mes_3'], errors='coerce')
enigh_final['mes_4'] = pd.to_numeric(enigh_final['mes_4'], errors='coerce')
enigh_final['mes_5'] = pd.to_numeric(enigh_final['mes_5'], errors='coerce')
enigh_final['mes_6'] = pd.to_numeric(enigh_final['mes_6'], errors='coerce')


enigh_final = enigh_final.drop_duplicates()
# Iterar sobre las filas del DataFrame
for index, row in enigh_final.iterrows():
    # Iterar sobre los índices de mes (1 a 6)
    for i in range(1, 7):
        # Obtener los nombres de las columnas mes_x e ing_x
        columna_mes = f'mes_{i}'
        columna_ingreso = f'ing_{i}'
        
        # Obtener los valores de mes_x e ing_x de la fila actual
        mes_x = row[columna_mes]
        ing_x = row[columna_ingreso]
        
        # Verificar si mes_x es igual a i
        if 2<= mes_x <=10:
            # Asignar el valor de ing_x a la columna con el nombre correspondiente i
            enigh_final.at[index, str(mes_x)] = ing_x

# Crear un diccionario de mapeo para los nuevos nombres de las columnas
nuevos_nombres = {'2.0': 'febrero', '3.0': 'marzo', '4.0': 'abril', '5.0': 'mayo', '6.0': 'junio', '7.0': 'julio',
                  '8.0': 'agosto', '9.0': 'septiembre', '10.0': 'octubre'                  
                  }

# Renombrar las columnas del DataFrame
enigh_final = enigh_final.rename(columns=nuevos_nombres)
for columna in enigh_final.columns:
    # Obtener los valores únicos y contar su frecuencia
    conteo_valores = enigh_final[columna].value_counts()
    
    # Imprimir el nombre de la columna y su conteo de valores
    print(f'Columna: {columna}')
    print(conteo_valores)
    print('\n')

enigh_final.to_csv('E:/Users/lucia/ProyectoInferencia/ENIGH/enigh_datos.csv')