import numpy as np
import matplotlib.pyplot as plt

gamma = 1/(np.pi) + 0.5
def KondoTemp(U):
    return U * np.sqrt(gamma/(2*U)) * np.exp(-np.pi*U/(8*gamma) + np.pi*gamma/(2*U))

print(KondoTemp(10))
# U = np.arange(0, 5, 0.1)
# T = KondoTemp(U)

# plt.plot(U, T)
# plt.show()
