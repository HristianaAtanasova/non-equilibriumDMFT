import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift

tmax = 10
dt = 0.01
t = np.arange(0, tmax, dt)
Cut = np.pi/dt
ft = np.arange(0, Cut, dt)
wmax = np.pi/(2*dt)
dw = np.pi/Cut
fw = np.arange(-wmax, wmax, dw)
w = np.arange(-7, 7, dw)

Gles = np.loadtxt("Green_les_U=0.0_T=1_t=10.out").view(complex)
Ggtr = np.loadtxt("Green_gtr_U=0.0_T=1_t=10.out").view(complex)

N = int(Cut/dt)
Gret = np.zeros(N+1, complex)
Gret[0:int(len(t))] = Ggtr + np.conj(Gles)
Gret = np.imag(Gret) + 1j*np.real(Gret)

fGret = fftshift(fft(Gret)) * dt/np.pi
# fGret = np.imag(fGret) + 1j*np.real(fGret)
a = int((N-len(w))/2)
b = int((N+len(w))/2)

# plt.figure(1)
# plt.subplot(211)
# plt.plot(w, np.real(fGret[a:b]), 'r--', w, np.imag(fGret[a:b]), 'b--')
plt.plot(w, np.imag(fGret[a:b]))
# plt.plot(fw, np.real(fGret), 'r--', fw, np.imag(fGret), 'b--')
plt.ylabel('A(w)')
plt.xlabel('w')
plt.grid()
# plt.subplot(212)
# plt.plot(t, np.real(Gret[0:int(len(t))]), 'r--', label='Green_ret_r')
# plt.plot(t, np.imag(Gret[0:int(len(t))]), 'b--', label='Green_ret_i')
# plt.legend()
# plt.grid()
plt.show()
