import matplotlib.pyplot as plt
import numpy as np

tmax = 12
dt = 0.005
t = np.arange(0, tmax, dt)
U = 2
T = 0.06
turn = 0.5
# Amplitude = np.arange(0.0,4.0,3.0)
Amplitude = np.arange(0.5,1.0,1.0)
# Amplitude = np.arange(2.0,2.5,0.5)

# l = 1.0

# U = np.zeros(3, int)
# U[0] = 18
# U[1] = 28
# U[2] = 42
# U = np.arange(1.0,1.5,0.5)
#
# Amplitude = np.zeros(3)
# Amplitude[0] = 16.0
# Amplitude[1] = 28.0
# Amplitude[2] = 42.0
# Amplitude[3] = 2.0
# Amplitude[4] = 3.0

init = 1


K = np.zeros((4, 4, len(Amplitude), len(t), len(t)), complex)
M = np.zeros((len(Amplitude), len(t)), complex)
doubleOCC = np.zeros((len(Amplitude), len(t)), complex)
Error = np.zeros((len(Amplitude), len(t)), float)

file_K = 'K_U={}_T={}_t={}_dt={}_A={}_lambda={}_i={}_f={}_test.out'

# A_ = 0
for A_ in range(len(Amplitude)):
    for f in range(4):
        # K[init, f, A_] = np.loadtxt(file_K.format(U,T,tmax,dt,0.0,Amplitude[A_],init,f)).view(complex)
        K[init, f, A_] = np.loadtxt(file_K.format(U,T,tmax,dt,Amplitude[A_],0.0,init,f)).view(complex)
        # K[init, f, A_] = np.loadtxt(file_K.format(Amplitude[A_],T,tmax,dt,2.0,0.0,init,f)).view(complex)


    # Up = K[init,1,:].diagonal() + K[init,3,:].diagonal()
    # Down = K[init,2,:].diagonal() + K[init,3,:].diagonal()
    # Tot_Pop = Up + Down

    M[A_] = (K[init,1,A_].diagonal() + K[init,3,A_].diagonal() - (K[init,2,A_].diagonal() + K[init,3,A_].diagonal())) / (1-2*K[init,3,A_].diagonal())

    # err = np.sum(K[init,:,:],0)
    Error[A_] = np.abs(1 - np.sum(K[init,:,A_],0).diagonal())

    doubleOCC[A_] = K[init,3,A_].diagonal()

# plt.plot(t, np.real(Up), 'r--', label='$init = {init}$'.format(init=init))
# plt.plot(t, np.real(Down), 'b--', label='$init = {init}$'.format(init=init))

# plt.plot(t, np.real(Tot_Pop), 'r--', t, np.imag(Tot_Pop), 'b--', label='$init = {init}$'.format(init=init))

# i = 0
# plt.plot(t, np.real(K[init,0,i].diagonal()), 'k', label = '| >')
# plt.plot(t, np.real(K[init,1,i].diagonal()), 'r', label = '|up>')
# plt.plot(t, np.real(K[init,2,i].diagonal()), 'y', label = '|down>')
# plt.plot(t, np.real(K[init,3,i].diagonal()), 'b', label = '|up,down>')
# #
# i = 1
# plt.plot(t, np.real(K[init,0,i].diagonal()), 'k--')
# plt.plot(t, np.real(K[init,1,i].diagonal()), 'r--')
# plt.plot(t, np.real(K[init,2,i].diagonal()), 'y--')
# plt.plot(t, np.real(K[init,3,i].diagonal()), 'b--')
# #
# i = 2
# plt.plot(t, np.real(K[init,0,i].diagonal()), 'k')
# plt.plot(t, np.real(K[init,1,i].diagonal()), 'r')
# plt.plot(t, np.real(K[init,2,i].diagonal()), 'y')
# plt.plot(t, np.real(K[init,3,i].diagonal()), 'b')

# plt.plot(t, np.real(Error[0]), 'r')

# plt.plot(t, np.real(M[0]), 'blue', label = 'U=16')
# plt.plot(t, np.real(M[1]), 'red', label = 'U=28')
# plt.plot(t, np.real(M[2]), 'lime', label = 'U=42')
# plt.plot(t, np.real(M[3]), 'navy', label = 'A=2.0')
# plt.plot(t, np.real(M[4]), 'gray', label = 'A=3.0')
# plt.plot(t, np.real(M[0]))


# plt.plot(t, np.real(doubleOCC[0]), 'blue', label = 'U=16')
# plt.plot(t, np.real(doubleOCC[1]), 'red', label = 'U=28')
# plt.plot(t, np.real(doubleOCC[2]), 'lime', label = 'U=42')
# plt.plot(t, np.real(doubleOCC[3]), 'navy', label = 'A=2.0')
# plt.plot(t, np.real(doubleOCC[4]), 'gray', label = 'A=3.0')
plt.plot(t, np.real(doubleOCC[0]))

# plt.ylabel('Populations(t), U = 4.0, A = 0.0, 0.5, 1.0, 2.0, 3.0, lambda = 0.0, dt = 0.005')
# plt.ylabel('M(t), A = 2.0, lambda = 0.0 , U = 16, 28, 42, w = 11')
# plt.ylabel('d(t), A = 2.0, lambda = 0.0 , U = 16, 28, 42, w = 11')

plt.xlabel('$t$')

plt.grid()
plt.legend()
plt.show()
