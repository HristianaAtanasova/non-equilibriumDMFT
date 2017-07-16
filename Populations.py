import matplotlib.pyplot as plt
import numpy as np

tmax = 4
dt = 0.01
t = np.arange(0, tmax, dt)
U = 4
T = 1
turn = 2
field = 1
F0 = 0
l = 0
# l = np.arange(0,4,1)

K = np.zeros((4, 4, 4, len(t), len(t)), complex)
file_K = 'K_U={}_T={}_t={}_dt={}_turn={}_lambda={}_i={}_f={}_test.out'

# for l_ in range(len(l)):
for i in range(1,2,1):
    for f in range(4):
        # K[i, f, l_] = np.loadtxt(file_K.format(U,T,tmax,dt,turn,l[l_],i,f)).view(complex)
        K[i, f, 1] = np.loadtxt(file_K.format(U,T,tmax,dt,turn,l,i,f)).view(complex)
init = 1

# Up = K[init,1,1].diagonal() + K[init,3,1].diagonal()
# Down = K[init,2,1].diagonal() + K[init,3,1].diagonal()
# Tot_Pop = Up + Down

# M = (Up - Down)/(1-2*K[init,3].diagonal())

err = np.sum(K[init,:,1],0)
Error = np.abs(1 - err.diagonal())

# plt.plot(t, np.real(Up), 'r--', label='$init = {init}$'.format(init=init))
# plt.plot(t, np.real(Down), 'b--', label='$init = {init}$'.format(init=init))

# plt.plot(t, np.real(Tot_Pop), 'r--', t, np.imag(Tot_Pop), 'b--', label='$init = {init}$'.format(init=init))

# plt.plot(t, np.real(K[init,0,0].diagonal()), 'k--', t, np.real(K[init,1,0].diagonal()), 'k--', t, np.real(K[init,2,0].diagonal()), 'k--', t, np.real(K[init,3,0].diagonal()), 'k--')
# plt.plot(t, np.real(K[init,0,1].diagonal()), 'b--', t,  np.real(K[init,1,1].diagonal()), 'b--', t, np.real(K[init,2,1].diagonal()), 'b--', t, np.real(K[init,3,1].diagonal()), 'b--')
# plt.plot(t, np.real(K[init,0,2].diagonal()), 'r--', t,  np.real(K[init,1,2].diagonal()), 'r--', t, np.real(K[init,2,2].diagonal()), 'r--', t, np.real(K[init,3,2].diagonal()), 'r--')
# plt.plot(t, np.real(K[init,0,3].diagonal()), 'y--', t,  np.real(K[init,1,3].diagonal()), 'y-', t, np.real(K[init,2,3].diagonal()), 'y--', t, np.real(K[init,3,3].diagonal()), 'y--')

#plt.plot(t, np.real(Error), 'r--', label='$U = {U}$'.format(U=U))
# plt.plot(t, np.real(M), 'r--', label='$U = {U}$'.format(U=U))

plt.ylabel('$Population(t), lambda = 0.0, 0.05, 0.1$')
plt.xlabel('$t$')

plt.grid()
plt.legend()
plt.show()
