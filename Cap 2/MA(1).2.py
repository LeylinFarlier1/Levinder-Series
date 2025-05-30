import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Let us presume that X and e have been equal to zero for every period, up until what
# we will call period t = 1, at which point X1 receive a one-time shock equal to one
 #unit, via u1. In other words, u1 = 1, u2 = 0, u3 = 0 and so forth. Let us trace out
 #the effects of this shock:

# Parámetros
phi = 0.75
u__t = [1, 0, 0, 0]  # Impulso unitario en t=1
n = len(u__t)

# Inicializar respuesta
e__t = np.zeros(n)

# Aplicar la fórmula e_t = u_t + phi * u_{t-1}
for t in range(n):
    e__t[t] = u__t[t]
    if t > 0:
        e__t[t] += phi * u__t[t-1]

print("Respuesta al impulso:", e__t)
plt.figure(figsize=(8, 4))
plt.stem(range(n), e__t, use_line_collection=True)
plt.title('Respuesta al impulso de MA(1)')
plt.xlabel('Tiempo')
plt.ylabel('Respuesta')
plt.tight_layout()
plt.show()