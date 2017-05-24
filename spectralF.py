import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift

tmax = 5
dt = 0.005
t = np.arange(0, tmax, dt)
Cut = np.pi/dt
ft = np.arange(0, Cut, dt)
wmax = np.pi/(2*dt)
dw = np.pi/Cut
fw = np.arange(-wmax, wmax, dw)
w = np.arange(-7, 7, dw)

Mles = np.zeros((2, 4, len(t), len(t)), complex) # timedep/fft, init, lower/upper time
Mgtr = np.zeros((2, 4, len(t), len(t)), complex)

les = 'Green_les_t={}_dt={}_i={}.out'
gtr = 'Green_gtr_t={}_dt={}_i={}.out'

les_fft = 'Green_fft_les_t={}_dt={}_i={}.out'
gtr_fft = 'Green_fft_gtr_t={}_dt={}_i={}.out'

for i in range(4):
    Mles[0, i] = np.loadtxt(les.format(tmax,dt,i)).view(complex)
    Mgtr[0, i] = np.loadtxt(gtr.format(tmax,dt,i)).view(complex)

    Mles[1, i] = np.loadtxt(les_fft.format(tmax,dt,i)).view(complex)
    Mgtr[1, i] = np.loadtxt(gtr_fft.format(tmax,dt,i)).view(complex)

i= 0
timedep = 1

Gles = Mles[timedep, i, len(t)-1, ::-1]
Ggtr = Mgtr[timedep, i, len(t)-1, ::-1]

N = int(Cut/dt)
Gret = np.zeros(N+1, complex)
Gret[0:int(len(t))] = Ggtr + np.conj(Gles)
# Gret[0:int(len(t))] = Ggtr

fGret = fftshift(fft(Gret)) * dt/np.pi
fGret = np.imag(fGret) + 1j*np.real(fGret)
a = int((N-len(w))/2)
b = int((N+len(w))/2)

# plt.plot(t, np.imag(Gles), 'b--', t, np.real(Gles), 'r--')
# plt.plot(t, np.imag(Gret[0:int(len(t))]), 'b--', t, np.real(Gret[0:int(len(t))]), 'r--')
plt.plot(w, np.imag(fGret[a:b]), 'b--', w, np.real(fGret[a:b]), 'r--')
plt.legend(loc='best')
plt.ylabel('A($\omega$)')
plt.xlabel('$\omega$')
plt.grid()
plt.show()
