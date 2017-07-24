import matplotlib.pyplot as plt
import numpy as np

tmax = 2
dt = 0.01
t = np.arange(0, tmax, dt)
U = 4.0
T = 0.1
turn = 0.5
Amplitude = np.arange(0.0,1.0,1.0)
# Amplitude = np.arange(1.0,1.5,1.0)
# Amplitude = np.arange(3.0,3.5,1.0)

# l = 1.0

# U = np.zeros(3, int)
# U[0] = 18
# U[1] = 28
# U[2] = 42
# U = np.arange(1.0,1.5,0.5)
#
# Amplitude = np.zeros(4)
# Amplitude[0] = 0.5
# Amplitude[1] = 1.0
# Amplitude[2] = 1.5
# Amplitude[3] = 2.0

init = 1


K = np.zeros((4, 4, len(Amplitude), len(t), len(t)), complex)
M = np.zeros((len(Amplitude), len(t)), complex)
doubleOCC = np.zeros((len(Amplitude), len(t)), complex)
Error = np.zeros((len(Amplitude), len(t)), float)

file_K = 'K_U={}_T={}_t={}_dt={}_A={}_lambda={}_i={}_f={}_test.out'

# A_ = 0
for A_ in range(len(Amplitude)):
    for f in range(4):
        # K[init, f, 0] = np.loadtxt(file_K.format(U,T,tmax,dt,0.0,0,init,f)).view(complex)
        K[init, f, A_] = np.loadtxt(file_K.format(U,T,tmax,dt,0.0,Amplitude[A_],init,f)).view(complex)


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

i = 0
plt.plot(t, np.real(K[init,0,i].diagonal()), 'k', label = '| >')
plt.plot(t, np.real(K[init,1,i].diagonal()), 'r', label = '|up>')
plt.plot(t, np.real(K[init,2,i].diagonal()), 'y', label = '|down>')
plt.plot(t, np.real(K[init,3,i].diagonal()), 'b', label = '|up,down>')

# i = 1
# plt.plot(t, np.real(K[init,0,i].diagonal()), 'k--')
# plt.plot(t, np.real(K[init,1,i].diagonal()), 'r--')
# plt.plot(t, np.real(K[init,2,i].diagonal()), 'y--')
# plt.plot(t, np.real(K[init,3,i].diagonal()), 'b--')
#
# i = 2
# plt.plot(t, np.real(K[init,0,i].diagonal()), 'k')
# plt.plot(t, np.real(K[init,1,i].diagonal()), 'r')
# plt.plot(t, np.real(K[init,2,i].diagonal()), 'y')
# plt.plot(t, np.real(K[init,3,i].diagonal()), 'b')

# plt.plot(t, np.real(Error[0]), 'r')

# plt.plot(t, np.real(M[0]), 'black', t, np.real(M[1]), 'royalblue', t, np.real(M[2]), 'lime', t, np.real(M[3]), 'gray', t, np.real(M[4]), 'k')
# plt.plot(t, np.real(M[0]), 'black', t, np.real(M[1]), 'royalblue', t, np.real(M[2]), 'lime', t, np.real(M[3]), 'gray')
# plt.plot(t, np.real(M[0]), 'g', t, np.real(M[1]), 'r', t, np.real(M[2]), 'b')
# plt.plot(t, np.real(M[0]))


# plt.plot(t, np.real(doubleOCC[0]), 'r', t, np.real(doubleOCC[1]), 'g', t, np.real(doubleOCC[2]), 'b', t, np.real(doubleOCC[3]), 'y', t, np.real(doubleOCC[4]), 'k')
# plt.plot(t, np.real(doubleOCC[0]), 'black', t, np.real(doubleOCC[1]), 'royalblue', t, np.real(doubleOCC[2]), 'lime', t, np.real(doubleOCC[len(Amplitude)-1]), 'gray')
# plt.plot(t, np.real(doubleOCC[0]), 'red', t, np.real(doubleOCC[1]), 'olive', t, np.real(doubleOCC[2]), 'blue', t, np.real(doubleOCC[3]), 'lime', t, np.real(doubleOCC[4]), 'black', t, np.real(doubleOCC[4]), 'yellow', t, np.real(doubleOCC[4]), 'plum')
# plt.plot(t, np.real(doubleOCC[0]), 'black', t, np.real(doubleOCC[1]), 'royalblue', t, np.real(doubleOCC[2]), 'lime', t, np.real(doubleOCC[3]), 'gray')
# plt.plot(t, np.real(doubleOCC[0]))

# plt.ylabel('Populations(t), U = 5, A = 0, lambda = 0.0, 1.0, 2.0')
# plt.ylabel('M(t), U = 2, A = 0, dt = 0.0025')
# plt.ylabel('d(t), A = 0.0, 0.5, .., 3.0, lambda = 0.0 , U = 40.0, w = 20')

plt.xlabel('$t$')

plt.grid()
plt.legend()
plt.show()
