import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import zipfile
import matplotlib.pyplot as plt

loaded = np.load('Green.npz')
t = loaded['t']
Green = loaded['Green']

# plt.plot(t,np.real(Green[0,0,:,0]),'r', t, np.imag(Green[0,0,:,0]),'b')
# plt.plot(t,np.real(Green[0,0,0,:]),'r--', t, np.imag(Green[0,0,0,:]),'b--')


dt = 0.01
# t = np.arange(0, 5, dt)
Cut = np.pi/dt
ft = np.arange(0, Cut, dt)
wmax = np.pi/dt
dw = np.pi/Cut
fw = np.arange(-wmax, wmax, dw)
w = np.arange(-10, 10, dw)

spin = 0
U = 8
T = 1

# Gles = 1j*Green[1, spin, -1, ::-1]
# Ggtr = -1j*Green[0, spin, :, 10]

Gles = 1j*Green[1, spin, len(t)-1, ::-1]
Ggtr = -1j*Green[0, spin, ::-1, len(t)-1]

N = int(Cut/dt)
Gadv = np.zeros(N+1, complex)
Gadv[0:int(len(t))] = (Gles - Ggtr)
# Gadv[0:int(len(t))] = -Ggtr
# Gadv[0:int(len(t))] = Gles

fGadv = fftshift(fft(Gadv)) * dt / (np.pi)
# fGadv = np.imag(fGadv) + 1j*np.real(fGadv)
a = int((N-len(w))/2)
b = int((N+len(w))/2)

number = 10
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0,1,20)]

# plt.plot(t, np.imag(Gles), 'b--', t, np.real(Gles), 'r--')
# plt.plot(t, np.imag(Ggtr), 'y--', t, np.real(Ggtr), 'k--')
# plt.plot(t, np.imag(Gadv[0:int(len(t))]), 'b--', t, np.real(Gadv[0:int(len(t))]), 'r--')
plt.plot(w, np.imag(fGadv[a:b]), color=colors[U], label='$U = {U}$'.format(U=U))
# plt.plot(w, np.imag(fGadv[a:b]), color=colors[t_], label='$t = {t_}$'.format(t_=t_))

plt.legend(loc='best')
# plt.title('U=4 | T=1')
# plt.ylabel('$error(t)$')
plt.ylabel('$A(\omega)$')
plt.xlabel('$\omega$')
plt.grid()
plt.show()
