import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def semicircularDos(w, v_0):
    """
    DOS for Bethe Lattice
    """
    return 1 / (2 * np.pi * v_0 ** 2) * np.sqrt(4 * v_0 ** 2 - w ** 2)

def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))

def genSemicircularHyb(T, mu, v_0, tmax, dt, dw):
    """
    Generate Hybridization function for Fermion bath
    """
    beta = 1.0 / T
    Cut  = np.pi / dt

    t    = np.arange(0, tmax, dt)
    wDOS = np.arange(-2 * v_0, 2 * v_0, dw)
    w    = np.arange(-Cut, Cut, dw)

    Delta = np.zeros((2, 2, len(t), len(t)), complex)  # indices are gtr/les | spin up/spin down

    # window function padded with zeros for semicircular DOS
    N = int(2*Cut/dw)
    a = int(N/2+2*v_0/dw)
    b = int(N/2-2*v_0/dw)
    DOS = np.zeros(N+1)
    DOS[b:a] = semicircularDos(wDOS, v_0)

    # frequency-domain Hybridization function
    Hyb_les = DOS * fermi_function(w, beta, mu)
    Hyb_gtr = DOS * (1 - fermi_function(w, beta, mu))

    fDelta_les = np.conj(ifftshift(fft(fftshift(Hyb_les)))) * dw/np.pi
    fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi

    # get real times from fft_times
    Delta_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down

    for t1 in range(len(t)):

        Delta_init[0, :, t1] = fDelta_gtr[int(N / 2) + t1]
        Delta_init[1, :, t1] = fDelta_les[int(N / 2) + t1]

    for t1 in range(len(t)):
        for t2 in range(len(t)):
            Delta[:, :, t1, t2] = tdiff(Delta_init[0, 0], t1, t2)

return Delta
