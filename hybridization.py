import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])


def hypercubicDos(w, v_0):
    """
    DOS for the cubic lattice in the limit d -> inf 
    """
    return np.exp(-(w ** 2) / v_0 ** 2) / (np.sqrt(np.pi) * v_0)


def semicircularDos(w, v_0):
    """
    DOS for Bethe Lattice
    """
    return 1 / (2 * np.pi * v_0 ** 2) * np.sqrt(4 * v_0 ** 2 - w ** 2)
    # return 2 / (np.pi * v_0) * np.sqrt(1 - (w/v_0)**2)


def flatBand(w, wC, v, gamma):
    """
    DOS for a flat band with soft cutoff
    """
    return gamma / ((np.exp(v * (w - wC)) + 1) * (np.exp(-v * (w + wC)) + 1))


def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))


def genGaussianHyb(T, mu, v_0, tmax, dt, wC, dw):
    """
    Generate Hybridization function for Fermion bath with a semicircular DOS
    """
    beta = 1.0 / T
    Cut  = np.pi / dt

    t = np.arange(0, tmax, dt)
    w = np.arange(-wC, wC, dw)
    fw = np.arange(-Cut, Cut, dw)

    Delta = np.zeros((2, 2, len(t), len(t)), complex)  # indices are gtr/les | spin up/spin down

    # window function padded with zeros for semicircular DOS
    N = len(fw)
    w_start = int(N / 2 - int(wC / dw))
    w_end = int(N /2 + int(wC / dw))
    dos = np.zeros(N)
    dos[w_start:w_end] = hypercubicDos(w, v_0)

    fermi = fermi_function(fw, beta, mu)

    # frequency-domain Hybridization function
    Hyb_les = dos * fermi 
    Hyb_gtr = dos * (1 - fermi)

    fDelta_les = ifftshift(fft(fftshift(Hyb_les))) * dw / np.pi
    fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw / np.pi

    # get real times from fft_times
    Delta_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down

    for t1 in range(len(t)):
        Delta_init[0, :, t1] = fDelta_gtr[int(N / 2) + t1]
        Delta_init[1, :, t1] = fDelta_les[int(N / 2) + t1]

    for t1 in range(len(t)):
        for t2 in range(len(t)):
            Delta[0, 0, t2, t1] = tdiff(Delta_init[0, 0], t2, t1)
            Delta[0, 1, t2, t1] = tdiff(Delta_init[0, 1], t2, t1)
            Delta[1, 0, t1, t2] = tdiff(Delta_init[1, 0], t2, t1)
            Delta[1, 1, t1, t2] = tdiff(Delta_init[1, 1], t2, t1)

    # plt.plot(w, dos[w_start:w_end], label = 'dos')
    # plt.plot(w, fermi[w_start:w_end], '-', w, (1-fermi[w_start:w_end]), '--', label = 'fermi')
    # plt.plot(w, np.real(Hyb_les[w_start:w_end]), '-', w, np.imag(Hyb_les[w_start:w_end]), '--', label = 'Hyb_les')
    # plt.plot(w, np.real(Hyb_gtr[w_start:w_end]), '-', w, np.imag(Hyb_gtr[w_start:w_end]), '--', label = 'Hyb_gtr')
    # plt.show()
    # # plt.savefig('delta.pdf')
    # plt.close()

    np.savez_compressed('Delta_mu={}_T={}_dt={}'.format(mu, T, dt), t=t, dos=dos[w_start:w_end], D=Delta)

def genSemicircularHyb(T, mu, v_0, wC, tmax, dt, dw):
    """
    Generate Hybridization function for Fermion bath with a gaussian DOS
    """
    beta = 1.0 / T
    Cut  = np.pi / dt

    t = np.arange(0, tmax, dt)
    w_semi = np.arange(-2.0 * v_0, 2.0 * v_0, dw)
    w = np.arange(-wC, wC, dw)
    fw = np.arange(-Cut, Cut, dw)

    # t    = np.arange(0, tmax, dt)
    # wDOS = np.arange(-2 * v_0, 2 * v_0, dw)
    # w    = np.arange(-Cut, Cut, dw)

    Delta = np.zeros((2, 2, len(t), len(t)), complex)  # indices are gtr/les | spin up/spin down

    N = len(fw)
    w_start = int(N / 2 - int(2.0 * v_0 / dw))
    w_end = int(N /2 + int(2.0 * v_0 / dw))
    dos = np.zeros(N)
    dos[w_start:w_end] = semicircularDos(w_semi, v_0)

    fermi = fermi_function(fw, beta, mu)

    # # window function padded with zeros for semicircular DOS
    # N = int(2 * Cut / dw)
    # a = int(N / 2 + 2 * v_0 / dw)
    # b = int(N / 2 - 2 * v_0 / dw)
    # DOS = np.zeros(N + 1)
    # DOS[b:a] = semicircularDos(wDOS, v_0)

    # frequency-domain Hybridization function
    Hyb_les = dos * fermi
    Hyb_gtr = dos * (1 - fermi)

    fDelta_les = ifftshift(fft(fftshift(Hyb_les))) * dw/np.pi
    fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi

    # get real times from fft_times
    Delta_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down

    for t1 in range(len(t)):
        Delta_init[0, :, t1] = fDelta_gtr[int(N / 2) + t1]
        Delta_init[1, :, t1] = fDelta_les[int(N / 2) + t1]

    for t1 in range(len(t)):
        for t2 in range(len(t)):
            Delta[0, 0, t2, t1] = tdiff(Delta_init[0, 0], t2, t1)
            Delta[0, 1, t2, t1] = tdiff(Delta_init[0, 1], t2, t1)
            Delta[1, 0, t1, t2] = tdiff(Delta_init[1, 0], t2, t1)
            Delta[1, 1, t1, t2] = tdiff(Delta_init[1, 1], t2, t1)

    w_start = int(N / 2 - int(wC / dw))
    w_end = int(N /2 + int(wC / dw))

    # plt.plot(w, dos[w_start:w_end], label = 'dos')
    # plt.savefig('dos.pdf')
    # plt.close()

    # plt.plot(w, np.real(Hyb_les[w_start:w_end]), '-', w, np.imag(Hyb_les[w_start:w_end]), '--', label = 'Hyb_les')
    # plt.plot(w, np.real(Hyb_gtr[w_start:w_end]), '-', w, np.imag(Hyb_gtr[w_start:w_end]), '--', label = 'Hyb_gtr')
    # plt.show()

    np.savez_compressed('Delta_mu={}_T={}_dt={}'.format(mu, T, dt), t=t, dos=dos[w_start:w_end], D=Delta)


def genWideBandHyb(T, mu, wC, tmax, dt, dw):
    """
    Generate Hybridization function for Fermion bath with a wide, flat band
    """
    # additional parameters to generate the flat band
    v = 1.0
    gamma = 1.0
    beta = 1.0 / T
    Cut  = np.pi / dt
    
    dw = dt
    t = np.arange(0, tmax, dt)
    w = np.arange(-wC, wC, dw)
    fw = np.arange(-Cut, Cut, dw)

    N = len(fw)
    w_start = int(N / 2 - int(wC / dw))
    w_end = int(N /2 + int(wC / dw))
    dos = np.zeros(N)

    dos = flatBand(fw, wC, v, gamma)
    fermi = fermi_function(fw, beta, mu)

    Delta = np.zeros((2, 2, len(t), len(t)), complex)  # indices are gtr/les | spin up/spin down

    # frequency-domain Hybridization function
    Hyb_les = dos  * fermi
    Hyb_gtr = dos * (1 - fermi)

    # fDelta_les = np.conj(ifftshift(fft(fftshift(Hyb_les)))) * dw/np.pi
    fDelta_les = ifftshift(fft(fftshift(Hyb_les))) * dw/np.pi
    fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi

    # get real times from fft_times
    Delta_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down

    for t1 in range(len(t)):
        Delta_init[0, :, t1] = fDelta_gtr[int(N / 2) + t1]
        Delta_init[1, :, t1] = fDelta_les[int(N / 2) + t1]

    for t1 in range(len(t)-1,0,-1):
        for t2 in range(len(t)-1,0,-1):
    # for t1 in range(len(t)):
    #     for t2 in range(len(t)):
            Delta[0, 0, t2, t1] = tdiff(Delta_init[0, 0], t2, t1)
            Delta[0, 1, t2, t1] = tdiff(Delta_init[0, 1], t2, t1)
            Delta[1, 0, t1, t2] = tdiff(Delta_init[1, 0], t2, t1)
            Delta[1, 1, t1, t2] = tdiff(Delta_init[1, 1], t2, t1)

    t_cut = len(t)
    # t_cut = 1

    # plt.plot(t[:t_cut], np.real(Delta_init[1, 1, :t_cut]), t[:t_cut], np.imag(Delta_init[1, 1, :t_cut]), '--', label='Delta_les')
    # plt.plot(t[:t_cut], np.real(Delta_init[0, 1, :t_cut]), t[:t_cut], np.imag(Delta_init[0, 1, :t_cut]), '--', label='Delta_gtr')   
    # plt.legend()
    # plt.show()


    # plt.plot(t[:t_cut], np.real(Delta[1, 1, :t_cut, t_cut-1]), t[:t_cut], np.imag(Delta[1, 1, :t_cut, t_cut-1]), '--', label='Delta_les')
    # plt.plot(t[:t_cut], np.real(Delta[1, 1, t_cut-1, :t_cut]), t[:t_cut], np.imag(Delta[1, 1, t_cut-1, :t_cut]), '--', label='Delta_gtr')   
    # plt.legend()
    # plt.savefig('Delta_mu={}_T={}_dt={}.pdf'.format(mu, T, dt))
    # plt.close()

    # plt.plot(fw[w_start:w_end], np.real(Hyb_les[w_start:w_end]), '-', fw[w_start:w_end], np.imag(Hyb_les[w_start:w_end]), '--', label = 'Hyb_les')
    # plt.plot(fw[w_start:w_end], np.real(Hyb_gtr[w_start:w_end]), '-', fw[w_start:w_end], np.imag(Hyb_gtr[w_start:w_end]), '--', label = 'Hyb_gtr')
    # plt.legend()
    # plt.savefig('Hybridization_mu={}_T={}_dt={}.pdf'.format(mu, T, dt))
    # plt.close()
    # # plt.show()

    np.savez_compressed('Delta_mu={}_T={}_dt={}'.format(mu, T, dt) , t=t, D=Delta)
