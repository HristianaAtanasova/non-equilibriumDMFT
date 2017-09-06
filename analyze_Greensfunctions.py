import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift

tmax = 2
dt = 0.005
t = np.arange(0, tmax, dt)
U = 15
T = 0.1
turn = 0.5
field = 1
F0 = 0
l = 0.05

Green = np.zeros((4, 2, 2, len(t), len(t)), complex) # initial state, gtr/les, spin up/spin down
Up = np.zeros((4, len(t)), complex)
Down = np.zeros((4, len(t)), complex)

gtr_up = 'gtr_up_U={}_T={}_t={}_dt={}_turn={}_lambda={}_i={}_test.out'
les_up = 'les_up_U={}_T={}_t={}_dt={}_turn={}_lambda={}_i={}_test.out'

gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}_turn={}_lambda={}_i={}_test.out'
les_down = 'les_down_U={}_T={}_t={}_dt={}_turn={}_lambda={}_i={}_test.out'

# gtr_up = 'gtr_up_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'
# les_up = 'les_up_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'
#
# gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'
# les_down = 'les_down_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}.out'

for i in range(1,2,1):
    Green[i, 0, 0] = np.loadtxt(gtr_up.format(U,T,tmax,dt,turn,l,i)).view(complex)
    Green[i, 1, 0] = np.loadtxt(les_up.format(U,T,tmax,dt,turn,l,i)).view(complex)
    Green[i, 0, 1] = np.loadtxt(gtr_down.format(U,T,tmax,dt,turn,l,i)).view(complex)
    Green[i, 1, 1] = np.loadtxt(les_down.format(U,T,tmax,dt,turn,l,i)).view(complex)


    Up[i] = Green[i,1,0].diagonal()
    Down[i] = Green[i,1,1].diagonal()

# Pop_Up = np.sum(Up,0)/4
# Pop_Down = np.sum(Down,0)/4

init = 1
Pop_Up = Up[init]
Pop_Down = Down[init]

Pop = Pop_Up + Pop_Down

# plt.plot(t, np.real(Pop), 'r--', label='init={}_U={}_dt={}'.format(init,U,dt))
# plt.plot(t, np.imag(Pop), 'b--')
plt.plot(t, np.real(Pop_Up), 'r--', label='init={}_U={}_dt={}'.format(init,U,dt))
plt.plot(t, np.real(Pop_Down), 'y--', label='init={}_U={}_dt={}'.format(init,U,dt))
# plt.plot(t, np.real(Green[init, 1, 0, len(t)-1, :len(t)]), 'r--', t, np.imag(Green[init, 1, 0, len(t)-1, :len(t)]), 'b--', label='Gles_spin_up')
# plt.plot(t, np.real(Green[init, 1, 1, len(t)-1, :len(t)]), 'y--', t, np.imag(Green[init, 1, 1, len(t)-1, :len(t)]), 'k--', label='Gles_spin_down')

plt.legend(loc='best')
plt.ylabel('G(t)')
plt.xlabel('($t$)')
plt.grid()
plt.show()

plt.show()
