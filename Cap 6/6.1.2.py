# ------------------------------------------------------------
# Notas: Diferenciación estacional (seasonal differencing)
# ------------------------------------------------------------

# Contexto:
# En análisis de datos mensuales de ventas minoristas, las ventas en diciembre 
# suelen ser las más altas del año debido a las festividades. Comparar diciembre 
# con noviembre no es útil (es obvio que diciembre > noviembre). 
# Lo relevante es comparar diciembre con diciembre del año anterior.

# Comparación correcta:
# - Comparar Xt con Xt-12 (mes actual vs mismo mes del año anterior)
# - Si las ventas de este diciembre son mayores que las del diciembre pasado:
#   Xt - Xt−12 > 0
#   También puede escribirse como: (1 - L^12)Xt > 0

# Diferenciación estacional:
# - Se resta la observación actual con la observación del mismo periodo en el año anterior.
# - Para datos trimestrales (seasonality trimestral): Xt − Xt−4
# - Para datos mensuales (seasonality mensual): Xt − Xt−12

# Implicación:
# Para datos no estacionarios con tendencia + estacionalidad:
# → Se requieren dos niveles de diferenciación:
#   1. Diferencia regular (Xt − Xt−1) para eliminar raíz unitaria de largo plazo
#   2. Diferencia estacional (Xt − Xt−s) para eliminar raíz unitaria estacional

# Notación:
# - Superscript (por ejemplo: D²) → diferencia de primer orden aplicada dos veces
# - Subscript (por ejemplo: D₄, D₁₂) → diferenciación estacional:
#     D₄X = (1 - L⁴)X = X - L⁴X    ← datos trimestrales
#     D₁₂X = (1 - L¹²)X = X - L¹²X ← datos mensuales

# Ejemplo con datos reales:
# - Dataset: tasa de desempleo trimestral (EE.UU., personas de 15 a 64 años)
# - temp en 1962q2 = tasa(1962q2) - tasa(1961q2) = 5.5 - 7.1 = -1.6
# - temp en 1962q1 = tasa(1962q1) - tasa(1961q1) = 6.5 - 8 = -1.5

# Gráfico:
# Se puede graficar la serie original y la diferenciada estacionalmente 
# para visualizar la remoción del componente estacional.

# Conclusión:
# El lag de diferenciación depende de la frecuencia de los datos y del tipo de estacionalidad.
# - Datos trimestrales → usar L=4
# - Datos mensuales → usar L=12

import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

# --- Configuración de las series a analizar ---
fred_series_codes_info = {
    'LRUN64TTUSQ156S': {
        'name': 'Tasa de Desempleo (15-64 años, EE.UU.)',
        'diff_period': 4,
        'ylabel': 'Tasa (%)'
    },
    'GDPC1': {
        'name': 'PIB Real (EE.UU.)',
        'diff_period': 4,
        'ylabel': 'Billones $ (2017)'
    },
    'CPIAUCSL': {
        'name': 'Índice de Precios al Consumidor (EE.UU.)',
        'diff_period': 12,
        'ylabel': 'Índice 1982-84=100'
    }
}

start_date = datetime.datetime(1980, 1, 1)
end_date = datetime.datetime.now()

# --- Crear la figura y los ejes para todos los subplots ---
num_series = len(fred_series_codes_info)
# Aumentamos un poco el tamaño vertical por serie
fig, axes = plt.subplots(num_series, 2, figsize=(15, 6 * num_series), sharex=False) 
fig.suptitle("Análisis de Estacionalidad de Series Económicas de FRED", fontsize=10)

row_idx = 0

# --- Bucle para procesar cada serie ---
for series_code, info in fred_series_codes_info.items():
    print(f"\n--- Procesando serie: {info['name']} ({series_code}) ---")
    
    try:
        data_df = web.DataReader(series_code, 'fred', start_date, end_date)
        print("Datos descargados exitosamente.")

        data_df.rename(columns={series_code: 'valor_original'}, inplace=True)

        diff_period = info['diff_period']
        data_df[f'D{diff_period}_valor'] = data_df['valor_original'].diff(diff_period)
        
        ax_original = axes[row_idx, 0]
        ax_original.plot(data_df.index, data_df['valor_original'], label=f"Original", color='blue')
        ax_original.set_title(f"{info['name']} Original")
        ax_original.set_xlabel('Fecha')
        ax_original.set_ylabel(info['ylabel'])
        ax_original.legend()
        ax_original.grid(True)
        
        ax_diferenciada = axes[row_idx, 1]
        ax_diferenciada.plot(data_df.index, data_df[f'D{diff_period}_valor'], label=f'Diferencia (D{diff_period})', color='orange')
        ax_diferenciada.set_title(f'Diferencia Estacional de {info["name"]}')
        ax_diferenciada.set_xlabel('Fecha')
        ax_diferenciada.set_ylabel(f'Dif. (D{diff_period})')
        ax_diferenciada.legend()
        ax_diferenciada.grid(True)
        
        row_idx += 1
        
    except Exception as e:
        print(f"Error al procesar la serie {series_code} ({info['name']}): {e}")
        if row_idx < num_series:
            axes[row_idx, 0].set_title(f"Error al cargar {info['name']}")
            axes[row_idx, 0].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center')
            axes[row_idx, 1].set_title(f"Error al cargar {info['name']}")
            axes[row_idx, 1].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center')
            row_idx += 1

# Ajustar el layout para evitar superposiciones y mostrar la figura
# plt.tight_layout(rect=[0, 0, 1, 0.96]) # tight_layout puede ser reemplazado o complementado por subplots_adjust
plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.05, hspace=0.4, wspace=0.3) # Ajustar hspace y wspace según sea necesario
plt.show()

print("\n--- Fin del análisis de todas las series ---")