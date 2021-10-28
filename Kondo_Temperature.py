
import numpy as np
# import matplotlib.pyplot as plt

gamma = 1/(np.pi) + 0.5
# gamma = np.sqrt(5) + 1/(np.pi)
def KondoTemp(U):
    return U * np.sqrt(gamma/(2*U)) * np.exp(-np.pi*U/(8*gamma) + np.pi*gamma/(2*U))

print(KondoTemp(5))
# U = np.arange(0, 5, 0.1)
# T = KondoTemp(U)

# plt.plot(U, T)
# plt.show()
