import yfinance as yf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
#descargar los datos de dow jones industrial index desde 2000 hasta 2010.
DJI = yf.download("^DJI", start="2000-01-01", end="2010-12-31")

DJI_diff= DJI.diff().dropna()  # Calcular la diferencia diaria

#grafico la seria diaria como su primeras diferencias
plot.figure(figsize=(12, 6))
plot.subplot(2, 1, 1)
plot.plot(DJI['Close'], label='Dow Jones Industrial Average', color='blue')
plot.title('Dow Jones Industrial Average (2000-2010)')
plot.xlabel('Fecha')
plot.ylabel('Precio de Cierre')
plot.legend()
plot.subplot(2, 1, 2)
plot.plot(DJI_diff['Close'], label='Diferencias Diarias', color='orange')
plot.title('Diferencias Diarias del Dow Jones Industrial Average')
plot.xlabel('Fecha')
plot.ylabel('Diferencia Diaria')
plot.legend()
plot.tight_layout()
plot.show()