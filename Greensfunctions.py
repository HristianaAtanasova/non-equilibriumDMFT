import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift

tmax = 2
dt = 0.005
t = np.arange(0, tmax, dt)


Gles_fft = np.zeros((4, len(t), len(t)), complex)
Ggtr_fft = np.zeros((4, len(t), len(t)), complex)

Gles = np.zeros((4, len(t), len(t)), complex)
Ggtr = np.zeros((4, len(t), len(t)), complex)

les = 'Green_les_t={}_dt={}_i={}.out'
gtr = 'Green_gtr_t={}_dt={}_i={}.out'

les_fft = 'Green_fft_les_t={}_dt={}_i={}.out'
gtr_fft = 'Green_fft_gtr_t={}_dt={}_i={}.out'

for i in range(4):
    Gles_fft[i] = np.loadtxt(les_fft.format(tmax,dt,i)).view(complex)
    Ggtr_fft[i] = np.loadtxt(gtr_fft.format(tmax,dt,i)).view(complex)

    Gles[i] = np.loadtxt(les.format(tmax,dt,i)).view(complex)
    Ggtr[i] = np.loadtxt(gtr.format(tmax,dt,i)).view(complex)

i= 0

plt.plot(t, np.real(Gles[i, len(t)-1, :]), 'r--', t, np.imag(Gles[i, len(t)-1, :]), 'r--', label='Gles')
# plt.plot(t, np.real(Gles_fft[i, :, 0]), 'b--', t, np.imag(Gles_fft[i, :, 0]), 'b--', label='Gles_fft')
plt.plot(t, np.real(Gles_fft[i, len(t)-1, :]), 'b--', t, np.imag(Gles_fft[i, len(t)-1, :]), 'b--', label='Gles_fft')
plt.legend(loc='best')
plt.ylabel('G(t)')
plt.xlabel('($t$)')
plt.grid()
plt.show()
