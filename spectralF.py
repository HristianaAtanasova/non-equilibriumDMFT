import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift

tmax = 5
dt = 0.01
t = np.arange(0, tmax, dt)
U = 8
T = 0.1
turn = 0.5
field = 1
F0 = 0
l = 1

Cut = np.pi/dt
ft = np.arange(0, Cut, dt)
wmax = np.pi/dt
dw = np.pi/Cut
fw = np.arange(-wmax, wmax, dw)
w = np.arange(-8, 8, dw)

Green = np.zeros((2, 2, len(t), len(t)), complex) # initial state, gtr/les, spin up/spin down

gtr_up = 'gtr_up_U={}_T={}_t={}_dt={}_turn={}_lambda={}_test.out'
les_up = 'les_up_U={}_T={}_t={}_dt={}_turn={}_lambda={}_test.out'

gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}_turn={}_lambda={}_test.out'
les_down = 'les_down_U={}_T={}_t={}_dt={}_turn={}_lambda={}_test.out'

# gtr_up = 'gtr_up_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'
# les_up = 'les_up_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'
#
# gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'
# les_down = 'les_down_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'

spin = 0

Green[0, 0] = np.loadtxt(gtr_up.format(U,T,tmax,dt,turn,l)).view(complex)
Green[1, 0] = np.loadtxt(les_up.format(U,T,tmax,dt,turn,l)).view(complex)
Green[0, 1] = np.loadtxt(gtr_down.format(U,T,tmax,dt,turn,l)).view(complex)
Green[1, 1] = np.loadtxt(les_down.format(U,T,tmax,dt,turn,l)).view(complex)

# Gles = 1j*np.sum(Green[1, :, len(t)-1, ::-1],0)/2
# Ggtr = -1j*np.sum(Green[0, :, len(t)-1, ::-1],0)/2

Gles = 1j*Green[1, spin, len(t)-1, ::-1]
Ggtr = -1j*Green[0, spin, len(t)-1, ::-1]

N = int(Cut/dt)
Gadv = np.zeros(N+1, complex)
# Gadv[0:int(len(t))] = (Gles - Ggtr)
Gadv[0:int(len(t))] = (Gles + np.conj(Ggtr))
# Gadv[0:int(len(t))] = -Ggtr
# Gadv[0:int(len(t))] = Gles


fGadv = fftshift(fft(Gadv)) / (np.pi)
# fGadv = np.imag(fGadv) + 1j*np.real(fGadv)
a = int((N-len(w))/2)
b = int((N+len(w))/2)

# plt.plot(t, np.imag(Gles), 'b--', t, np.real(Gles), 'r--')
# plt.plot(t, np.imag(Ggtr), 'y--', t, np.real(Ggtr), 'k--')
# plt.plot(t, np.imag(Gadv[0:int(len(t))]), 'b--', t, np.real(Gadv[0:int(len(t))]), 'r--')
plt.plot(w, np.imag(fGadv[a:b]), 'b--', w, np.real(fGadv[a:b]), 'r--')
plt.legend(loc='best')
plt.ylabel('A($\omega$)')
plt.xlabel('$\omega$')
plt.grid()
plt.show()
