#Suppose we estimated the following AR(3) model
 #Xt =0.75Xt−1 +0.50Xt−2 +0.10Xt−3 +et.
 #(2.11)
 #Suppose further that X1 = 5, X2 =−10 and X3 = 15
 #Consider the model described in Eq.(2.11). Forecast from period 4 to 10

# Datos iniciales
X = [5, -10, 15]  # Valores iniciales X1, X2, X3
b1 = 0.75  # Coeficiente AR(1)
b2 = 0.50  # Coeficiente AR(2)
b3 = 0.10  # Coeficiente AR(3)

# Número de periodos a predecir (del periodo 4 al 10)
n_pred = 7

# Realizar las predicciones
for t in range(n_pred):
    Xt = b1 * X[-1] + b2 * X[-2] + b3 * X[-3]
    X.append(Xt)

# Imprimir resultados
for i in range(3, 10):
    print(f"X_{i+1} = {X[i]:.2f}")
    
from matplotlib import pyplot as plt
# Graficar los valores predichos
plt.figure(figsize=(10, 4))
plt.plot(range(1, 11), X, marker='o', label='Predicciones AR(3)')
plt.title('Predicciones del Modelo AR(3)')
plt.xlabel('Periodo')
plt.ylabel('Valor de X')
plt.xticks(range(1, 11))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


#vemos que diverge pq no es estacionaria   