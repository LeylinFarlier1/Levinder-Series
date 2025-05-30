# Teoría del Random Walk:
# El modelo Random Walk se define como Xt = Xt−1 + et, donde et es ruido blanco (media 0, varianza σ²).
# Aplicando sustituciones sucesivas:
# Xt = X0 + e1 + e2 + ... + et
# 
# Esperanza condicional: E(Xt | X0) = X0, ya que E(et | X0) = 0 ∀ t.
# Esto implica que el mejor pronóstico de Xt en el tiempo t = 0 es simplemente X0.
#
# Varianza:
# Var(Xt) = Var(X0 + e1 + e2 + ... + et)
# Como los errores son independientes y X0 es constante:
# Var(Xt) = σ² + σ² + ... + σ² = t * σ²
# La varianza depende del tiempo ⇒ el proceso no es estacionario en varianza.
#
# Diferenciación:
# Sea Zt = Xt − Xt−1 ⇒ Zt = et
# Zt es ruido blanco (media constante, varianza constante, no autocorrelación) ⇒ proceso estacionario.
#
# Conclusión:
# Un Random Walk es un proceso I(1), no estacionario, pero su primera diferencia es I(0), es decir, estacionaria.

n = 100
import numpy as np
e_t = np.random.normal(0, 1, n)  # Ruido blanco

X_t = np.zeros(n)
for t in range(1, n):
    X_t[t] = X_t[t - 1] + e_t[t]  # Random Walk
    
#calculo la primera diferencia
Z_t = np.diff(X_t)  # Primera diferencia, no produce NaN

#graficar la serie original y la primera diferencia
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(X_t, label='Serie Original (Random Walk)', color='blue')
plt.title('Serie Original (Random Walk)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.axhline(0, color='red', linestyle='--', label='Media Condicional Teorica = 0')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Z_t, label='Primera Diferencia (Ruido Blanco)', color='orange')
plt.title('Primera Diferencia (Ruido Blanco)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.axhline(0, color='red', linestyle='--', label='Media = 0')
plt.legend()
plt.tight_layout()
plt.show()