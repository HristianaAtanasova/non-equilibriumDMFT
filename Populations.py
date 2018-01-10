import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import zipfile
import matplotlib.pyplot as plt

loaded = np.load('K_1_f.npz')
t = loaded['t']
K = loaded['K']

plt.plot(t,np.real(K[0].diagonal()),'k', label='|0>')
plt.plot(t,np.real(K[1].diagonal()),'r', label='|up>')
plt.plot(t,np.real(K[2].diagonal()),'y', label='|down>')
plt.plot(t,np.real(K[3].diagonal()),'b', label='|up,down>')

# P = K[0].diagonal() + K[1].diagonal() + K[2].diagonal() + K[3].diagonal()
# plt.plot(t,np.real(P))

plt.legend(loc='best')
plt.title('U=4 | T=1')
plt.ylabel('$Populations(t)$')
plt.xlabel('$t$')
plt.grid()
plt.show()
