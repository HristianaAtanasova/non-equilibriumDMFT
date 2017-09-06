import matplotlib.pyplot as plt
import numpy as np

for U in range(0,5,1):
    tmax = 6
    dt = 0.005
    t = np.arange(0, tmax, dt)
    # U = 2
    T = 1
    init = 1

    # U = np.zeros(3, int)
    # U[0] = 18
    # U[1] = 28
    # U[2] = 42
    # U = np.arange(1.0,1.5,0.5)
    #
    Amplitude = np.arange(0.0,4.0,5.0)
    # Amplitude = np.arange(0.5,1.0,1.0)
    # Amplitude = np.arange(2.0,2.5,0.5)
    #
    # Amplitude = np.zeros(4)
    # Amplitude[0] = 1
    # Amplitude[1] = 0.5
    # Amplitude[2] = 0.1
    # Amplitude[3] = 0.05
    # Amplitude[4] = 3.0

    K = np.zeros((4, 4, len(Amplitude), len(t), len(t)), complex)
    M = np.zeros((len(Amplitude), len(t)), complex)
    doubleOCC = np.zeros((len(Amplitude), len(t)), complex)
    Error = np.zeros((len(Amplitude), len(t)), float)

    # file_K = 'K_U={}.0_T={}_t={}_dt={}_A={}_lambda={}_i={}_f={}.out'
    file_K = 'K_U={}.0_T={}_t={}_dt={}_A={}_lambda={}_i={}_f={}_ph.out'

    for A in range(len(Amplitude)):
        for f in range(4):
            # K[init, f, A] = np.loadtxt(file_K.format(U,T,tmax,dt,0.0,Amplitude[A],init,f)).view(complex)
            K[init, f, A] = np.loadtxt(file_K.format(U,T,tmax,dt,Amplitude[A],0,init,f)).view(complex)
            # K[init, f, A] = np.loadtxt(file_K.format(Amplitude[A],T,tmax,dt,2.0,0.0,init,f)).view(complex)


        # Up = K[init,1,:].diagonal() + K[init,3,:].diagonal()
        # Down = K[init,2,:].diagonal() + K[init,3,:].diagonal()
        # Tot_Pop = Up + Down

        # M[A_] = (K[init,1,A_].diagonal() + K[init,3,A_].diagonal() - (K[init,2,A_].diagonal() + K[init,3,A_].diagonal())) / (1-2*K[init,3,A_].diagonal())

        # err = np.sum(K[init,:,:],0)
        Error[A] = np.abs(1 - np.sum(K[init,:,A],0).diagonal())

        # doubleOCC[A_] = K[init,3,A_].diagonal()

        # plt.plot(t, np.real(Up), 'r--', label='$init = {init}$'.format(init=init))
        # plt.plot(t, np.real(Down), 'b--', label='$init = {init}$'.format(init=init))
        # plt.plot(t, np.real(Tot_Pop), 'r--', t, np.imag(Tot_Pop), 'b--', label='$init = {init}$'.format(init=init))

        # plt.plot(t, np.real(K[init,0,A].diagonal()), 'k', label='| >, $Gamma = {A}$'.format(A=Amplitude[A]))
        # plt.plot(t, np.real(K[init,1,A].diagonal()), 'r', label = '|up>, $Gamma = {A}$'.format(A=Amplitude[A]))
        # plt.plot(t, np.real(K[init,2,A].diagonal()), 'y', label = '|down>, $Gamma = {A}$'.format(A=Amplitude[A]))
        # plt.plot(t, np.real(K[init,3,A].diagonal()), 'b', label = '|up,down>, $Gamma = {A}$'.format(A=Amplitude[A]))


        number = 10
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0,1,10)]

        plt.plot(t, np.real(Error[0]), color=colors[U], label='$U = {U}/D$'.format(U=U))
        # plt.plot(t, np.real(Error[A]), color=colors[A], label='$Gamma = {A}$'.format(A=Amplitude[A]))

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
        # plt.plot(t, np.real(doubleOCC[0]))

        # plt.ylabel('Populations(t), U = 4.0, A = 0.0, 0.5, 1.0, 2.0, 3.0, lambda = 0.0, dt = 0.005')
        # plt.ylabel('M(t), A = 2.0, lambda = 0.0 , U = 16, 28, 42, w = 11')
        # plt.ylabel('d(t), A = 2.0, lambda = 0.0 , U = 16, 28, 42, w = 11')

plt.xlabel('$t$')
plt.grid()
plt.legend()
plt.show()
