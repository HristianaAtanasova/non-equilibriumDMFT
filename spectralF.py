import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift


for U in range(0,5,1):

    tmax = 6
    dt = 0.005
    t = np.arange(0, tmax, dt)
    # U = 4
    T = 1
    l = 0
    A = 0.0
    spin = 0

    Green = np.zeros((2, 2, len(t), len(t)), complex) # initial state, gtr/les, spin up/spin down

    gtr_up = 'gtr_up_U={}.0_T={}_t={}_dt={}_A={}_lambda={}_ph.out'
    les_up = 'les_up_U={}.0_T={}_t={}_dt={}_A={}_lambda={}_ph.out'

    gtr_down = 'gtr_down_U={}.0_T={}_t={}_dt={}_A={}_lambda={}_ph.out'
    les_down = 'les_down_U={}.0_T={}_t={}_dt={}_A={}_lambda={}_ph.out'

    # gtr_up = 'gtr_up_U={}.0_T={}_t={}_dt={}_A={}_lambda={}.out'
    # les_up = 'les_up_U={}.0_T={}_t={}_dt={}_A={}_lambda={}.out'
    #
    # gtr_down = 'gtr_down_U={}.0_T={}_t={}_dt={}_A={}_lambda={}.out'
    # les_down = 'les_down_U={}.0_T={}_t={}_dt={}_A={}_lambda={}.out'

    Green[0, 0] = np.loadtxt(gtr_up.format(U,T,tmax,dt,A,l)).view(complex)
    Green[1, 0] = np.loadtxt(les_up.format(U,T,tmax,dt,A,l)).view(complex)
    Green[0, 1] = np.loadtxt(gtr_down.format(U,T,tmax,dt,A,l)).view(complex)
    Green[1, 1] = np.loadtxt(les_down.format(U,T,tmax,dt,A,l)).view(complex)

    for t_ in range(6,7,2):
        t = np.arange(0, t_, dt)
        Cut = np.pi/dt
        ft = np.arange(0, Cut, dt)
        wmax = np.pi/dt
        dw = np.pi/Cut
        fw = np.arange(-wmax, wmax, dw)
        w = np.arange(-8, 8, dw)

        Gles = 1j*Green[1, spin, len(t)-1, len(t)-1::-1]
        Ggtr = -1j*Green[0, spin, len(t)-1, len(t)-1::-1]

        N = int(Cut/dt)
        Gadv = np.zeros(N+1, complex)
        # Gadv[0:int(len(t))] = (Ggtr - Gles)
        Gadv[0:int(len(t))] = (Ggtr + np.conj(Gles))
        # Gadv[0:int(len(t))] = Ggtr
        # Gadv[0:int(len(t))] = -Gles

        fGadv = -fftshift(fft(Gadv)) / (np.pi)
        # fGadv = np.imag(fGadv) + 1j*np.real(fGadv)
        a = int((N-len(w))/2)
        b = int((N+len(w))/2)

        number = 10
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0,1,20)]

        # plt.plot(t, np.imag(Gles), 'b--', t, np.real(Gles), 'r--')
        # plt.plot(t, np.imag(Ggtr), 'y--', t, np.real(Ggtr), 'k--')
        # plt.plot(t, np.imag(Gadv[0:int(len(t))]), 'b--', t, np.real(Gadv[0:int(len(t))]), 'r--')
        plt.plot(w, np.imag(fGadv[a:b]), color=colors[U], label='$U = {U}/D$'.format(U=U))
        # plt.plot(w, np.imag(fGadv[a:b]), color=colors[t_], label='$t = {t_}$'.format(t_=t_))


plt.legend(loc='best')
plt.ylabel('A($\omega$)')
plt.xlabel('$\omega$')
plt.grid()
plt.show()
