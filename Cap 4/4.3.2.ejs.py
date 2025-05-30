#En un grafica anade estas dos lineas ∣β2∣< 1 . B2 es el eje de la y , e b1 el eje de la x
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#

#grafico las lineas en -1 y 1
#lue

beta2 = np.linspace(-5, 5, 400)
beta1 = np.linspace(-5, 5, 400)
B1, B2 = np.meshgrid(beta1, beta2)

# Condiciones para el área de estabilidad
region = (np.abs(B2) < 1) & (B2 > -B1**2 / 4) & (B2 < 1 - B1) & (B2 < 1 + B1)

# Nueva región: debajo de la parábola pero encima de -1
debajo_parabola = (B2 < -B1**2 / 4) & (B2 > -1) & (np.abs(B2) < 1)

plt.figure(figsize=(12, 8))

# Rellenar la región de estabilidad
plt.contourf(B1, B2, region, levels=[0.5, 1], colors=['#ffe082'], alpha=0.7, zorder=0)

# Rellenar la región debajo de la parábola pero encima de -1
plt.contourf(B1, B2, debajo_parabola, levels=[0.5, 1], colors=['#90caf9'], alpha=0.7, zorder=1)

plt.axhline(1, color='red', linestyle='--', label=r'$|\beta_2| < 1$')
plt.axhline(-1, color='red', linestyle='--')
plt.plot(beta1, 1 - beta1, color='blue', label=r'$\beta_2 = 1 - \beta_1$')
plt.plot(beta1, 1 + beta1, color='blue', label=r'$\beta_2 = 1 + \beta_1$')

# Agregar la parábola beta2 = -beta1**2 / 4
parabola_beta2 = -beta1**2 / 4
plt.plot(beta1, parabola_beta2, color='green', label=r'$\beta_2 = -\beta_1^2 / 4$')

plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.title('Condiciones de Estabilidad para AR(2)')
plt.legend()
plt.grid()
plt.show()


## En este gráfico, la región sombreada representa las condiciones de estabilidad para un proceso AR(2).
# La línea roja horizontal indica los límites de |β2| < 1, y las líneas azules muestran las restricciones lineales.
# La parábola verde representa la condición cuadrática para la estabilidad.
# En este gráfico, la región sombreada representa las condiciones de estabilidad para un proceso AR(2).

#Voy a crear varias series AR(2) estacionarias y no estacionarias

 #1. Xt = 1.10Xt−1 +et
 #2. Yt = 0.70Yt−1 +0.10Yt−2 +et
 #3. Zt = 0.80Zt−1 +0.30Zt−2 +et
 #4. Wt =−0.80Wt−1 +0.30Wt−2 +e
 
import numpy as np
import matplotlib.pyplot as plt
# Configuración de la semilla para reproducibilidad
np.random.seed(42)
# Número de observaciones
n = 100
# Coeficientes para las series AR(2)
# Series originales
beta1_1 = 1.10
beta2_1 = 0.0  # No hay segundo término, por lo que beta2 es 0
beta1_2 = 0.70
beta2_2 = 0.10
beta1_3 = 0.80
beta2_3 = 0.30
beta1_4 = -0.80
beta2_4 = 0.30

# Series de los ejercicios
beta1_a = 1.05
beta2_a = 0.0
beta1_b = 0.60
beta2_b = 0.10
beta1_c = 0.50
beta2_c = 0.30
beta1_d = 0.80
beta2_d = -0.10
# Generar series AR(2)
def generar_serie_ar2(beta1, beta2, n):
    serie = np.zeros(n)
    for t in range(2, n):
        et = np.random.normal(0, 1)  # Ruido blanco
        serie[t] = beta1 * serie[t - 1] + beta2 * serie[t - 2] + et
    return serie
# Generar las series
serie_1 = generar_serie_ar2(beta1_1, beta2_1, n)
serie_2 = generar_serie_ar2(beta1_2, beta2_2, n)
serie_3 = generar_serie_ar2(beta1_3, beta2_3, n)
serie_4 = generar_serie_ar2(beta1_4, beta2_4, n)

# Series de los ejercicios
serie_a = generar_serie_ar2(beta1_a, beta2_a, n)
serie_b = generar_serie_ar2(beta1_b, beta2_b, n)
serie_c = generar_serie_ar2(beta1_c, beta2_c, n)
serie_d = generar_serie_ar2(beta1_d, beta2_d, n)
# Graficar las series en un solo subplot (8 filas, 1 columna)
fig, axs = plt.subplots(8, 1, figsize=(14, 20), sharex=True)

# Series originales
# Cambia el layout a 4 filas y 2 columnas
fig, axs = plt.subplots(4, 2, figsize=(16, 16), sharex=True)
axs = axs.flatten()
axs[0].plot(serie_1, marker='o', linestyle='-', color='blue', label='Serie 1: Xt = 1.10Xt−1 + et')
axs[0].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[0].set_ylabel('Valor')
axs[0].legend()

axs[1].plot(serie_2, marker='o', linestyle='-', color='orange', label='Serie 2: Yt = 0.70Yt−1 + 0.10Yt−2 + et')
axs[1].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[1].set_ylabel('Valor')
axs[1].legend()

axs[2].plot(serie_3, marker='o', linestyle='-', color='green', label='Serie 3: Zt = 0.80Zt−1 + 0.30Zt−2 + et')
axs[2].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[2].set_ylabel('Valor')
axs[2].legend()

axs[3].plot(serie_4, marker='o', linestyle='-', color='purple', label='Serie 4: Wt = -0.80Wt−1 + 0.30Wt−2 + et')
axs[3].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[3].set_ylabel('Valor')
axs[3].legend()

# Series de los ejercicios
axs[4].plot(serie_a, marker='o', linestyle='-', color='cyan', label='Serie a: Xt = 1.05Xt−1 + et')
axs[4].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[4].set_ylabel('Valor')
axs[4].legend()

axs[5].plot(serie_b, marker='o', linestyle='-', color='magenta', label='Serie b: Yt = 0.60Yt−1 + 0.10Yt−2 + et')
axs[5].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[5].set_ylabel('Valor')
axs[5].legend()

axs[6].plot(serie_c, marker='o', linestyle='-', color='brown', label='Serie c: Zt = 0.50Zt−1 + 0.30Zt−2 + et')
axs[6].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[6].set_ylabel('Valor')
axs[6].legend()

axs[7].plot(serie_d, marker='o', linestyle='-', color='olive', label='Serie d: Wt = 0.80Wt−1 - 0.10Wt−2 + et')
axs[7].axhline(0, color='red', linestyle='--', label='Media = 0')
axs[7].set_xlabel('Tiempo')
axs[7].set_ylabel('Valor')
axs[7].legend()

plt.tight_layout()
plt.show()

#calculo los modulos de los polinomios característicos
from statsmodels.tsa.ar_model import AutoReg
# Crear DataFrames para las series
df_serie_1 = pd.DataFrame({'serie': serie_1})
df_serie_2 = pd.DataFrame({'serie': serie_2})
df_serie_3 = pd.DataFrame({'serie': serie_3})
df_serie_4 = pd.DataFrame({'serie': serie_4})

df_serie_a = pd.DataFrame({'serie': serie_a})
df_serie_b = pd.DataFrame({'serie': serie_b})
df_serie_c = pd.DataFrame({'serie': serie_c})
df_serie_d = pd.DataFrame({'serie': serie_d})

# Ajustar modelos AR(2)
modelo_serie_1 = AutoReg(df_serie_1['serie'], lags=2).fit()
modelo_serie_2 = AutoReg(df_serie_2['serie'], lags=2).fit()
modelo_serie_3 = AutoReg(df_serie_3['serie'], lags=2).fit()
modelo_serie_4 = AutoReg(df_serie_4['serie'], lags=2).fit()

modelo_serie_a = AutoReg(df_serie_a['serie'], lags=2).fit()
modelo_serie_b = AutoReg(df_serie_b['serie'], lags=2).fit()
modelo_serie_c = AutoReg(df_serie_c['serie'], lags=2).fit()
modelo_serie_d = AutoReg(df_serie_d['serie'], lags=2).fit()
# Obtener las raíces de los polinomios característicos
raices_serie_1 = modelo_serie_1.roots
raices_serie_2 = modelo_serie_2.roots
raices_serie_3 = modelo_serie_3.roots
raices_serie_4 = modelo_serie_4.roots
# Calcular los módulos de las raíces
modulos_serie_1 = np.abs(raices_serie_1)
modulos_serie_2 = np.abs(raices_serie_2)
modulos_serie_3 = np.abs(raices_serie_3)
modulos_serie_4 = np.abs(raices_serie_4)

# Imprimir los módulos de las raíces
print("Módulos de las raíces de la Serie 1 (Xt = 1.10Xt−1 + et):", modulos_serie_1)
print("Módulos de las raíces de la Serie 2 (Yt = 0.70Yt−1 + 0.10Yt−2 + et):", modulos_serie_2)
print("Módulos de las raíces de la Serie 3 (Zt = 0.80Zt−1 + 0.30Zt−2 + et):", modulos_serie_3)
print("Módulos de las raíces de la Serie 4 (Wt = -0.80Wt−1 + 0.30Wt−2 + et):", modulos_serie_4)

#grafico las lineas en -1 y 1
#lue

beta2 = np.linspace(-5, 5, 400)
beta1 = np.linspace(-5, 5, 400)
B1, B2 = np.meshgrid(beta1, beta2)

# Condiciones para el área de estabilidad
region = (np.abs(B2) < 1) & (B2 > -B1**2 / 4) & (B2 < 1 - B1) & (B2 < 1 + B1)

# Nueva región: debajo de la parábola pero encima de -1
debajo_parabola = (B2 < -B1**2 / 4) & (B2 > -1) & (np.abs(B2) < 1)

plt.figure(figsize=(12, 8))

# Rellenar la región de estabilidad
plt.contourf(B1, B2, region, levels=[0.5, 1], colors=['#ffe082'], alpha=0.7, zorder=0)

# Rellenar la región debajo de la parábola pero encima de -1
plt.contourf(B1, B2, debajo_parabola, levels=[0.5, 1], colors=['#90caf9'], alpha=0.7, zorder=1)

plt.axhline(1, color='red', linestyle='--', label=r'$|\beta_2| < 1$')
plt.axhline(-1, color='red', linestyle='--')
plt.plot(beta1, 1 - beta1, color='blue', label=r'$\beta_2 = 1 - \beta_1$')
plt.plot(beta1, 1 + beta1, color='blue', label=r'$\beta_2 = 1 + \beta_1$')

# Agregar la parábola beta2 = -beta1**2 / 4
parabola_beta2 = -beta1**2 / 4
plt.plot(beta1, parabola_beta2, color='green', label=r'$\beta_2 = -\beta_1^2 / 4$')

# Añadir los módulos de las raíces como puntos
# (Asegúrate de que los módulos ya estén calculados antes de este bloque)
plt.scatter([beta1_1], [beta2_1], color='black', s=80, marker='o', label='Serie 1')
plt.scatter([beta1_2], [beta2_2], color='orange', s=80, marker='o', label='Serie 2')
plt.scatter([beta1_3], [beta2_3], color='green', s=80, marker='o', label='Serie 3')
plt.scatter([beta1_4], [beta2_4], color='purple', s=80, marker='o', label='Serie 4')

plt.scatter([beta1_a], [beta2_a], color='cyan', s=80, marker='o', label='Serie a')
plt.scatter([beta1_b], [beta2_b], color='magenta', s=80, marker='o', label='Serie b')
plt.scatter([beta1_c], [beta2_c], color='brown', s=80, marker='o', label='Serie c')
plt.scatter([beta1_d], [beta2_d], color='olive', s=80, marker='o', label='Serie d')

plt.xlabel(r'$\beta_1$')
plt.ylabel(r'$\beta_2$')
plt.title('Condiciones de Estabilidad para AR(2)')
plt.legend()
plt.grid()
plt.show()