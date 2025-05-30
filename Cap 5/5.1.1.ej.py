# Análisis de series temporales: Orden de integración
# =============================================================================
# Objetivo:
# - Descargar datos macroeconómicos (PIB per cápita nominal y el IPC de EE. UU.)
# - Calcular primera y segunda diferencia
# - Graficar la serie original, la primera y segunda diferencia
# - Identificar visualmente el posible orden de integración (I(0), I(1), I(2))
#
# Definiciones:
# - Una serie es estacionaria (I(0)) si su media, varianza y autocovarianza
#   no cambian en el tiempo.
# - Si no es estacionaria pero su primera diferencia sí lo es, entonces es I(1).
# - Si requiere dos diferenciaciones para volverse estacionaria, es I(2).

# =============================================================================
# EJERCICIO: Análisis del orden de integración de series macroeconómicas
# =============================================================================
# Para cada una de las siguientes series:
# 
# (a) PIB per cápita nominal de EE. UU. 
#     - Indicador: ny.gdp.pcap.cd
#     - Comando Stata: 
#         wbopendata, country(usa) indicator(ny.gdp.pcap.cd) 
#         year(1960:2015) long clear
#
# (b) Índice de Precios al Consumidor (CPI) de EE. UU. 
#     - Indicador: fp.cpi.totl
#     - Comando Stata: 
#         wbopendata , country(usa) indicator(fp.cpi.totl) 
#         year(1960:2015) long clear
#
# Se debe realizar lo siguiente:
# 1. Descargar los datos desde el Banco Mundial.
# 2. Calcular la primera diferencia de la serie (ΔX = Xt - Xt−1).
# 3. Calcular la segunda diferencia de la serie (Δ²X = ΔXt - ΔXt−1).
# 4. Graficar:
#    - La serie original
#    - La primera diferencia
#    - La segunda diferencia
# 5. Identificar visualmente el posible orden de integración de cada serie
#    (es decir, cuántas diferenciaciones se requieren para que sea estacionaria).
# =============================================================================
# For each of the items listed below, you should be able to do the following:
# Download the data, and calculate the first and second differences. 
# Graph the original series and the two differenced series. 
# Visually identify its possible order of integration.
#
# (a) The nominal US GDP per capita. The command to download the data into
# Stata is:
# wbopendata, country(usa) indicator(ny.gdp.pcap.cd)
# year(1960:2015) long clear.
#
# (b) US CPI. The command to download the data into Stata is:
# wbopendata , country(usa) indicator(fp.cpi.totl)
# year(1960:2015) long clear

#DEscargar los datos desde el Banco Mundial
import pandas as pd
import matplotlib.pyplot as plt
import wbdata
# Configurar el indicador y el país
indicadores = {
    'ny.gdp.pcap.cd': 'PIB per cápita nominal (USD)',
    'fp.cpi.totl': 'Índice de Precios al Consumidor (CPI)'
}
# Definir el país (EE. UU.)
pais = 'USA'
pais1= 'Argentina'

from datetime import datetime
# Definir el rango de años
anios = (datetime(1960, 1, 1), datetime(2015, 12, 31))
# Descargar los datos
datos = wbdata.get_dataframe(indicadores, country=pais, date=anios)
# Convertir el índice a tipo datetime
datos.index = pd.to_datetime(datos.index)
# Calcular la primera diferencia
datos['crecimiento pbi per capita'] = datos['PIB per cápita nominal (USD)'].diff()
datos['inflacion'] = datos['Índice de Precios al Consumidor (CPI)'].pct_change() * 100

# Calcular la segunda diferencia
datos['Aceleración pbi per capita'] = datos['crecimiento pbi per capita'].diff()
datos['Aceleración inflación'] = datos['inflacion'].diff()
# Graficar las series originales y las diferencias
plt.figure(figsize=(14, 10))
plt.subplot(3, 2, 1)
plt.plot(datos.index, datos['PIB per cápita nominal (USD)'], label='PIB per cápita nominal (USD)', color='blue')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.subplot(3, 2, 2)
plt.plot(datos.index, datos['Índice de Precios al Consumidor (CPI)'], label='Índice de Precios al Consumidor (CPI)', color='orange')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.subplot(3, 2, 3)
plt.plot(datos.index, datos['crecimiento pbi per capita'], label='Crecimiento PIB per cápita', color='green')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.subplot(3, 2, 4)
plt.plot(datos.index, datos['inflacion'], label='Inflación (CPI)', color='red')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.subplot(3, 2, 5)
plt.plot(datos.index, datos['Aceleración pbi per capita'], label='Aceleración PIB per cápita', color='purple')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.subplot(3, 2, 6)
plt.plot(datos.index, datos['Aceleración inflación'], label='Aceleración Inflación (CPI)', color='brown')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================