import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift

tmax = 4
dt = 0.01
t = np.arange(0, tmax, dt)
Cut = np.pi/dt
ft = np.arange(0, Cut, dt)
wmax = np.pi/(2*dt)
dw = np.pi/Cut
fw = np.arange(-wmax, wmax, dw)
w = np.arange(-7, 7, dw)

Gles = np.zeros((4, len(t), len(t)), complex)
Ggtr = np.zeros((4, len(t), len(t)), complex)

# les = 'Green_les_t={}_dt={}_i={}.out'
# gtr = 'Green_gtr_t={}_dt={}_i={}.out'

les = 'Green_fft_les_t={}_dt={}_i={}.out'
gtr = 'Green_fft_gtr_t={}_dt={}_i={}.out'

for i in range(4):
    Gles[i] = np.loadtxt(les.format(tmax,dt,i)).view(complex)
    Ggtr[i] = np.loadtxt(gtr.format(tmax,dt,i)).view(complex)

i= 0
Gles = Gles[i, len(t)-1, ::-1]
Ggtr = Ggtr[i, len(t)-1, ::-1]

N = int(Cut/dt)
Gret = np.zeros(N+1, complex)
Gret[0:int(len(t))] = Ggtr + np.conj(Gles)

fGret = fftshift(fft(Gret)) * dt/np.pi
fGret = np.imag(fGret) + 1j*np.real(fGret)
a = int((N-len(w))/2)
b = int((N+len(w))/2)

plt.plot(w, np.imag(fGret[a:b]))
plt.legend(loc='best')
plt.ylabel('A($\omega$)')
plt.xlabel('$\omega$')
plt.grid()
plt.show()
