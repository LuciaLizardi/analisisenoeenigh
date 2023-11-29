# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:00:47 2023

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

enigh_final['ing_1'] = pd.to_numeric(enigh_final['ing_1'], errors='coerce')
enigh_final['ing_2'] = pd.to_numeric(enigh_final['ing_2'], errors='coerce')
enigh_final['ing_3'] = pd.to_numeric(enigh_final['ing_3'], errors='coerce')
enigh_final['ing_4'] = pd.to_numeric(enigh_final['ing_4'], errors='coerce')
enigh_final['ing_5'] = pd.to_numeric(enigh_final['ing_5'], errors='coerce')
enigh_final['ing_6'] = pd.to_numeric(enigh_final['ing_6'], errors='coerce')
enigh_final = enigh_final.drop_duplicates()


print(enigh_final.describe())

media_1 = enigh_final['ing_1'].mean()
print("Media Ingresos Mes 1 2022",media_1)

media_2 = enigh_final['ing_2'].mean()
print("Media Ingresos Mes 2 2022",media_2)

media_3 = enigh_final['ing_3'].mean()
print("Media Ingresos Mes 3 2022",media_3)

media_4 = enigh_final['ing_4'].mean()
print("Media Ingresos Mes 4 2022",media_4)

media_5 = enigh_final['ing_5'].mean()
print("Media Ingresos Mes 5 2022",media_5)

media_6 = enigh_final['ing_6'].mean()
print("Media Ingresos Mes 6 2022",media_6)



# Crear el gráfico de dispersión
enigh_final['ing_1'].plot.box(showfliers=False)
# Agregar etiquetas y título
plt.ylabel('Ingresos Mes 1')
# Mostrar el gráfico
plt.show()

enigh_final['ing_2'].plot.box(showfliers=False,vert=False)
# Agregar etiquetas y título
plt.ylabel('Ingresos Mes 2')
# Mostrar el gráfico
plt.show()

enigh_final['ing_3'].plot.box(showfliers=False,vert=False)
# Agregar etiquetas y título
plt.ylabel('Ingresos Mes 3')
# Mostrar el gráfico
plt.show()


enigh_final['ing_4'].plot.box(showfliers=False,vert=False)
# Agregar etiquetas y título
plt.ylabel('Ingresos Mes 4')
# Mostrar el gráfico
plt.show()

enigh_final['ing_5'].plot.box(showfliers=False,vert=False)
# Agregar etiquetas y título
plt.ylabel('Ingresos Mes 5')
# Mostrar el gráfico
plt.show()

enigh_final['ing_6'].plot.box(showfliers=False,vert=False)
# Agregar etiquetas y título
plt.ylabel('Ingresos Mes 6')
# Mostrar el gráfico
plt.show()

# Datos de ejemplo
data = pd.DataFrame({
    'Ingresos Mes 1':enigh_final['ing_1'],
    'Ingresos Mes 2':enigh_final['ing_2'],
    'Ingresos Mes 3':enigh_final['ing_3'],
    'Ingresos Mes 4':enigh_final['ing_4'],
    'Ingresos Mes 5':enigh_final['ing_5'],
    'Ingresos Mes 6':enigh_final['ing_6'],
})


# Crear un gráfico de tres boxplots horizontalmente
data.plot.box(vert=False,showfliers=False)
plt.xlabel('Ingresos')
plt.title('Diagramas de Caja y Brazos de Ingresos de Población >= 60 Años 2022')

#Definir nuevos datos para histogramas
data_1=enigh_final['ing_1']

data_2=enigh_final['ing_2']
data_2 = data_2[~np.isnan(data_2)]

data_3=enigh_final['ing_3']
data_3 = data_3[~np.isnan(data_3)]

data_4=enigh_final['ing_4']
data_4 = data_4[~np.isnan(data_4)]

data_5=enigh_final['ing_5']
data_5 = data_5[~np.isnan(data_5)]

data_6=enigh_final['ing_6']
data_6 = data_6[~np.isnan(data_6)]



# Calcular el rango intercuartílico (IQR) y definir límites para identificar datos atípicos
Q1_1, Q3_1 = np.percentile(data_1, [25, 75])
IQR_1 = Q3_1 - Q1_1
lower_bound_1 = Q1_1 - 1.5 * IQR_1
upper_bound_1 = Q3_1 + 1.5 * IQR_1
filtered_data_1 = data_1[(data_1 >= lower_bound_1) & (data_1 <= upper_bound_1)]
plt.figure(figsize=(8, 6))
sns.histplot(filtered_data_1, color='green', label='Mes 1', element='step', linewidth=2)

Q1_2, Q3_2 = np.percentile(data_2, [25, 75])
IQR_2 = Q3_2 - Q1_2
lower_bound_2 = Q1_2 - 1.5 * IQR_2
upper_bound_2 = Q3_2 + 1.5 * IQR_2
filtered_data_2 = data_2[(data_2 >= lower_bound_2) & (data_2 <= upper_bound_2)]
sns.histplot(filtered_data_2, color='blue', label='Mes 2', element='step', linewidth=2)

Q1_3, Q3_3 = np.percentile(data_3, [25, 75])
IQR_3 = Q3_3 - Q1_3
lower_bound_3 = Q1_3 - 1.5 * IQR_3
upper_bound_3 = Q3_3 + 1.5 * IQR_3
filtered_data_3 = data_3[(data_3 >= lower_bound_3) & (data_3 <= upper_bound_3)]
sns.histplot(filtered_data_3, color='red', label='Mes 3', element='step', linewidth=2)

Q1_4, Q3_4 = np.percentile(data_4, [25, 75])
IQR_4 = Q3_4 - Q1_4
lower_bound_4 = Q1_4 - 1.5 * IQR_4
upper_bound_4 = Q3_4 + 1.5 * IQR_4
filtered_data_4 = data_4[(data_4 >= lower_bound_4) & (data_4 <= upper_bound_4)]
sns.histplot(filtered_data_4,  color='orange', label='Mes 4', element='step', linewidth=2)

Q1_5, Q3_5 = np.percentile(data_5, [25, 75])
IQR_5 = Q3_5 - Q1_5
lower_bound_5 = Q1_5 - 1.5 * IQR_5
upper_bound_5 = Q3_5 + 1.5 * IQR_5
filtered_data_5 = data_5[(data_5 >= lower_bound_5) & (data_5 <= upper_bound_5)]
sns.histplot(filtered_data_5, color='yellow', label='Mes 5', element='step', linewidth=2)

Q1_6, Q3_6 = np.percentile(data_6, [25, 75])
IQR_6 = Q3_6 - Q1_6
lower_bound_6 = Q1_6 - 1.5 * IQR_6
upper_bound_6 = Q3_6 + 1.5 * IQR_6
filtered_data_6 = data_6[(data_6 >= lower_bound_6) & (data_6 <= upper_bound_6)]
sns.histplot(filtered_data_6,  color='pink', label='Mes 6', element='step', linewidth=2)

print("Data 1",filtered_data_1.describe())
var_1 = statistics.variance(filtered_data_1)

print("Data 2",filtered_data_2.describe())
var_2 = statistics.variance(filtered_data_2)

print("Data 3",filtered_data_3.describe())
var_3 = statistics.variance(filtered_data_3)

print("Data 4",filtered_data_4.describe())
var_4 = statistics.variance(filtered_data_4)

print("Data 5",filtered_data_5.describe())
var_5 = statistics.variance(filtered_data_5)

print("Data 6",filtered_data_6.describe())
var_6 = statistics.variance(filtered_data_6)

print("Varianza Mes 1",var_1)
print("Varianza Mes 2",var_2)
print("Varianza Mes 3",var_3)
print("Varianza Mes 4",var_4)
print("Varianza Mes 5",var_5)
print("Varianza Mes 6",var_6)


# Colores para cada línea de histograma
colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']

# Crear subgráficos para cada histograma
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, ax in enumerate(axes, start=1):
    # Obtener el DataFrame correspondiente
    current_data = globals()[f'data_{i}']  # asumiendo que tus DataFrames tienen nombres data_1, data_2, etc.

    # Asegurarse de que current_data sea un DataFrame
    if isinstance(current_data, pd.Series):
        current_data = current_data.to_frame()

    # Calcular el rango intercuartílico (IQR) y definir límites para identificar datos atípicos
    column_data = current_data[f'ing_{i}']
    Q1, Q3 = np.percentile(column_data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

    # Crear un histograma para cada columna con líneas y colores personalizados
    #sns.histplot(filtered_data, kde=True, color=colors[i-1], element='step', linewidth=2, ax=ax)
    sns.kdeplot(filtered_data, color=colors[i-1], ax=ax)

    # Añadir etiquetas y título
    ax.set_xlabel('Ingresos')
    ax.set_title(f'Histograma Ingresos Población >= 60 años Mes {i}, 2022')

# Ajustar el diseño de los subgráficos
plt.tight_layout()
plt.show()



ingresos = filtered_data_1
#ingresos = filtered_data_2
#ingresos = filtered_data_3
#ingresos = filtered_data_4
#ingresos = filtered_data_5
#ingresos = filtered_data_6


# Ajustar una distribución chi-cuadrado a los datos
df, loc, scale = chi2.fit(ingresos)

# Crear el histograma
plt.hist(ingresos, bins=30, density=True, alpha=0.6, color='yellow')

# Crear la función de densidad de probabilidad (PDF) ajustada
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = chi2.pdf(x, df, loc, scale)
plt.plot(x, p, 'k', linewidth=2)

title = "MES 6 2022 \n Ajuste de una Distribución Chi-cuadrado:\n grados de libertad $= %.2f$" % df
plt.title(title)

plt.show()


# Asumiendo que ingresos es una lista o array con datos de ingresos
ingresos = filtered_data_1


# Mostrar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

# Grados de libertad
df = 166326

# Valor de t para el percentil 0.975 y los grados de libertad especificados
valor_t_975 = t.ppf(0.975, df)

# Imprimir el resultado
print(f"El valor de t para el percentil 0.975 con {df} grados de libertad es aproximadamente: {valor_t_975}")


# Parámetros de la distribución chi-cuadrada original
grados_libertad = 1
tamaño_muestra = 166327

# Simular datos de la distribución chi-cuadrada
datos_chi_cuadrada = filtered_data_1

# Calcular la media muestral para cada muestra
medias_muestrales = np.mean(datos_chi_cuadrada)

# Parámetros de la distribución normal aproximada
media_aproximada = np.mean(medias_muestrales)
desviacion_estandar_aproximada = np.std(medias_muestrales)

# Imprimir la media y la desviación estándar de la distribución normal aproximada
print(f'Media muestral de la distribución normal aproximada: {media_aproximada}')
print(f'Desviación estándar muestral de la distribución normal aproximada: {desviacion_estandar_aproximada}')

# Visualizar la distribución normal aproximada
plt.hist(medias_muestrales, bins=10, density=True, alpha=0.6, color='red')
plt.title('Distribución Normal Aproximada de la Media Muestral Mes 1')
plt.xlabel('Ingresos Poblacion >=60 años')
plt.ylabel('Frecuencia relativa')
plt.show()




# Simular datos de la distribución chi-cuadrada
datos_todos =filtered_data_1.append(filtered_data_2, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_3, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_4, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_5, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_6, ignore_index=True)
print(datos_todos.describe())
# Calcular la media muestral para cada muestra
medias_muestrales = np.mean(datos_todos)

# Parámetros de la distribución normal aproximada
media_aproximada = np.mean(medias_muestrales)
desviacion_estandar_aproximada = np.std(medias_muestrales)

# Imprimir la media y la desviación estándar de la distribución normal aproximada
print(f'Media muestral de la distribución normal aproximada: {media_aproximada}')
print(f'Desviación estándar muestral de la distribución normal aproximada: {desviacion_estandar_aproximada}')

# Visualizar la distribución normal aproximada
plt.hist(medias_muestrales, bins=10, density=True, alpha=0.6, color='black')
plt.title('Distribución Normal Aproximada de la Media Muestral 2022')
plt.xlabel('Ingresos Mensuales Poblacion >=60 años')
plt.ylabel('Frecuencia relativa')
plt.show()