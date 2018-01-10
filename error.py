import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import zipfile
import matplotlib.pyplot as plt

loaded = np.load('K_1_f.npz')

t = loaded['t']
K = loaded['K']

Error = np.abs(1 - np.real(np.sum(K,0).diagonal()))

# P = K[0].diagonal() + K[1].diagonal() + K[2].diagonal() + K[3].diagonal()
# plt.plot(t,np.real(P))

# plt.plot(t,Error)

plt.legend(loc='best')
plt.title('U=4 | T=1')
plt.ylabel('$error(t)$')
plt.xlabel('$t$')
plt.grid()
plt.show()
