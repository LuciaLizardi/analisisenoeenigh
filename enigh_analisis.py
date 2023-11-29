# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:57:11 2023

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

datos_enigh=pd.read_csv('E:/Users/lucia/ProyectoInferencia/ENIGH/enigh_datos.csv')

print(datos_enigh.describe())

m_feb = datos_enigh['febrero'].mean()
print("Media Ingresos Febrero 2022",m_feb)

m_marz = datos_enigh['marzo'].mean()
print("Media Ingresos Marzo 2022",m_marz)

m_abril = datos_enigh['abril'].mean()
print("Media Ingresos Abril 2022",m_abril)

m_mayo =datos_enigh['mayo'].mean()
print("Media Ingresos Mayo 2022",m_mayo)

m_jun = datos_enigh['junio'].mean()
print("Media Ingresos Junio 2022",m_jun)

m_jul =datos_enigh['julio'].mean()
print("Media Ingresos Julio 2022",m_jul)

m_ago = datos_enigh['agosto'].mean()
print("Media Ingresos Agosto 2022",m_ago)

m_sept = datos_enigh['septiembre'].mean()
print("Media Ingresos Septiembre 2022",m_sept)

m_oct = datos_enigh['octubre'].mean()
print("Media Ingresos Octubre 2022",m_oct)


data = pd.DataFrame({
    'Ingresos Octubre ':datos_enigh['octubre'],
    'Ingresos Septiembre ':datos_enigh['septiembre'],
    'Ingresos Agosto ':datos_enigh['agosto'],
    'Ingresos Julio ':datos_enigh['julio'],
    'Ingresos Junio ':datos_enigh['junio'],
    'Ingresos Mayo ':datos_enigh['mayo'],    
    'Ingresos Abril ':datos_enigh['abril'],
    'Ingresos Marzo ':datos_enigh['marzo'],
    'Ingresos Febrero ':datos_enigh['febrero'],

})


# Crear un gráfico de tres boxplots horizontalmente
data.plot.box(vert=False,showfliers=False)
plt.xlabel('Ingresos')
plt.title('Diagramas de Caja y Brazos de Ingresos de Población >= 60 Años 2022')



data_2=datos_enigh['febrero']
data_2 = data_2[~np.isnan(data_2)]

data_3=datos_enigh['marzo']
data_3 = data_3[~np.isnan(data_3)]

data_4=datos_enigh['abril']
data_4 = data_4[~np.isnan(data_4)]

data_5=datos_enigh['mayo']
data_5 = data_5[~np.isnan(data_5)]

data_6=datos_enigh['junio']
data_6 = data_6[~np.isnan(data_6)]

data_7=datos_enigh['julio']
data_7 = data_7[~np.isnan(data_7)]

data_8=datos_enigh['agosto']
data_8 = data_8[~np.isnan(data_8)]

data_9=datos_enigh['septiembre']
data_9 = data_9[~np.isnan(data_9)]

data_10=datos_enigh['octubre']
data_10 = data_10[~np.isnan(data_10)]



#Calcular el rango intercuartílico (IQR) y definir límites para identificar datos atípicos
Q1_1, Q3_1 = np.percentile(data_2, [25, 75])
IQR_1 = Q3_1 - Q1_1
lower_bound_1 = Q1_1 - 1.5 * IQR_1
upper_bound_1 = Q3_1 + 1.5 * IQR_1
filtered_data_2 = data_2[(data_2 >= lower_bound_1) & (data_2 <= upper_bound_1)]
plt.figure(figsize=(8, 6))

Q1_2, Q3_2 = np.percentile(data_3, [25, 75])
IQR_2 = Q3_2 - Q1_2
lower_bound_2 = Q1_2 - 1.5 * IQR_2
upper_bound_2 = Q3_2 + 1.5 * IQR_2
filtered_data_3 = data_3[(data_3 >= lower_bound_2) & (data_3 <= upper_bound_2)]

Q1_4, Q3_4 = np.percentile(data_4, [25, 75])
IQR_4 = Q3_4 - Q1_4
lower_bound_4 = Q1_4 - 1.5 * IQR_4
upper_bound_4 = Q3_4 + 1.5 * IQR_4
filtered_data_4 = data_4[(data_4 >= lower_bound_4) & (data_4 <= upper_bound_4)]

Q1_5, Q3_5 = np.percentile(data_5, [25, 75])
IQR_5 = Q3_5 - Q1_5
lower_bound_5 = Q1_5 - 1.5 * IQR_5
upper_bound_5 = Q3_5 + 1.5 * IQR_5
filtered_data_5 = data_5[(data_5 >= lower_bound_5) & (data_5 <= upper_bound_5)]

Q1_6, Q3_6 = np.percentile(data_6, [25, 75])
IQR_6 = Q3_6 - Q1_6
lower_bound_6 = Q1_6 - 1.5 * IQR_6
upper_bound_6 = Q3_6 + 1.5 * IQR_6
filtered_data_6 = data_6[(data_6 >= lower_bound_6) & (data_6 <= upper_bound_6)]

Q1_7, Q3_7 = np.percentile(data_7, [25, 75])
IQR_7 = Q3_7 - Q1_7
lower_bound_7 = Q1_7 - 1.5 * IQR_7
upper_bound_7 = Q3_7 + 1.5 * IQR_7
filtered_data_7 = data_7[(data_7 >= lower_bound_7) & (data_7 <= upper_bound_7)]

Q1_8, Q3_8 = np.percentile(data_8, [25, 75])
IQR_8 = Q3_8 - Q1_8
lower_bound_8 = Q1_8 - 1.5 * IQR_8
upper_bound_8 = Q3_8 + 1.5 * IQR_8
filtered_data_8 = data_8[(data_8 >= lower_bound_8) & (data_8 <= upper_bound_8)]

Q1_9, Q3_9 = np.percentile(data_9, [25, 75])
IQR_9 = Q3_9 - Q1_9
lower_bound_9 = Q1_9 - 1.5 * IQR_9
upper_bound_9 = Q3_9 + 1.5 * IQR_9
filtered_data_9 = data_9[(data_9 >= lower_bound_9) & (data_9 <= upper_bound_9)]

Q1_10, Q3_10 = np.percentile(data_10, [25, 75])
IQR_10 = Q3_10 - Q1_10
lower_bound_10 = Q1_10 - 1.5 * IQR_10
upper_bound_10 = Q3_10 + 1.5 * IQR_10
filtered_data_10 = data_10[(data_10 >= lower_bound_10) & (data_10 <= upper_bound_10)]




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

print("Data 7",filtered_data_7.describe())
var_7 = statistics.variance(filtered_data_7)

print("Data 8",filtered_data_8.describe())
var_8 = statistics.variance(filtered_data_8)

print("Data 9",filtered_data_9.describe())
var_9 = statistics.variance(filtered_data_9)

print("Data 10",filtered_data_10.describe())
var_10 = statistics.variance(filtered_data_10)


print("Varianza Mes 2",var_2)
print("Varianza Mes 3",var_3)
print("Varianza Mes 4",var_4)
print("Varianza Mes 5",var_5)
print("Varianza Mes 6",var_6)
print("Varianza Mes 7",var_7)
print("Varianza Mes 8",var_8)
print("Varianza Mes 9",var_9)
print("Varianza Mes 10",var_10)


# Colores para cada línea de histograma
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']

# Crear subgráficos para cada histograma
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

# Iterar sobre los índices de las columnas "febrero", "marzo", ..., "octubre"
for i, mes in enumerate(['febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre'], start=1):
    # Obtener el DataFrame correspondiente
    current_data = globals()[f'data_{i+1}']  # asumiendo que tus DataFrames tienen nombres data_1, data_2, etc.

    # Asegurarse de que current_data sea un DataFrame
    if isinstance(current_data, pd.Series):
        current_data = current_data.to_frame()

    # Calcular el rango intercuartílico (IQR) y definir límites para identificar datos atípicos
    column_data = current_data[mes]
    Q1, Q3 = np.percentile(column_data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

    # Crear un histograma para cada columna con líneas y colores personalizados
    sns.kdeplot(filtered_data, color=colors[i-1], ax=axes[i-1])

    # Añadir etiquetas y título
    axes[i-1].set_xlabel(f'Ingresos {mes.capitalize()}')
    axes[i-1].set_title(f'Población >= 60 años {mes.capitalize()}, 2022')

# Ajustar el diseño de los subgráficos
plt.tight_layout()
plt.show()



#CHI
#ingresos = filtered_data_2
#ingresos = filtered_data_3
#ingresos = filtered_data_4
#ingresos = filtered_data_5
#ingresos = filtered_data_6
ingresos = filtered_data_10

# Ajustar una distribución chi-cuadrado a los datos
df, loc, scale = chi2.fit(ingresos)

# Crear el histograma
plt.hist(ingresos, bins=30, density=True, alpha=0.6, color='pink')

# Crear la función de densidad de probabilidad (PDF) ajustada
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = chi2.pdf(x, df, loc, scale)
plt.plot(x, p, 'k', linewidth=2)

title = "Octubre 2022 \n Ajuste de una Distribución Chi-cuadrado:\n grados de libertad $= %.2f$" % df
plt.title(title)

plt.show()

# Simular datos de la distribución chi-cuadrada
datos_todos =filtered_data_2.append(filtered_data_3, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_4, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_6, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_8, ignore_index=True)
datos_todos =datos_todos.append(filtered_data_10, ignore_index=True)


print(datos_todos.describe())
media=datos_todos.mean()
var = statistics.variance(datos_todos)
varianza= np.sqrt(var)/np.sqrt(568165)

x = np.linspace(media - 3 * np.sqrt(varianza), media + 3 * np.sqrt(varianza), 1000)
y = norm.pdf(x, media, np.sqrt(varianza))

# Grafica la distribución normal
plt.plot(x, y, label='Distribución Normal',color='black')

# Agrega etiquetas y título
plt.title('Distribución Normal')
plt.xlabel('Ingresos Mensuales Población >= 60 años')
plt.ylabel('Densidad de Probabilidad')
plt.legend()

# Muestra la gráfica
plt.show()


#Valor de fn de densidad t

# Parámetros
df = 568164
  # grados de libertad
scale = 1  # escala

# Puntos en el eje x
x = np.linspace(-5, 5, 1000)

# Calcular la función de densidad de probabilidad
pdf_values = t.pdf(x, df, scale)


# Graficar la función de densidad de probabilidad
plt.plot(x, pdf_values, label=f'Distribución t ({df} grados de libertad)')

# Etiquetas y título
plt.title('Función de Densidad de Probabilidad - Distribución t')
plt.xlabel('Valor')
plt.ylabel('Densidad de Probabilidad')
plt.legend()

# Mostrar la gráfica
plt.show()

percentil_975 = t.ppf(0.975, df)





df = 7

# Puntos en el eje x
x = np.linspace(0, 20, 1000)

# Calcular la función de densidad de probabilidad
pdf_values = chi2.pdf(x, df)

# Graficar la función de densidad de probabilidad
plt.plot(x, pdf_values, label=f'Chi-cuadrado ({df} grados de libertad)',color="gray")

# Etiquetas y título
plt.title('Función de Densidad de Probabilidad - Distribución chi-cuadrado')
plt.xlabel('Ingresos Mensuales Población >= 60 años (x1000)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()

# Mostrar la gráfica
plt.show()


