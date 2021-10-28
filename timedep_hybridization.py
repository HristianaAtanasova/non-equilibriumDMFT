import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.integrate import quad

import matplotlib.pyplot as plt

def tune_mu(t, mu):
    return mu / (1 + np.exp(-20 * (t - 1)))

def flatBand(w, wC, v, gamma):
    """
    DOS for a flat band with soft cutoff
    """
    return gamma / ((np.exp(v * (w - wC)) + 1) * (np.exp(-v * (w + wC)) + 1))

def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))

def genWideBandHyb(T, mu, tmax, dt, dw):
    """
    Generate Hybridization function for Fermion bath with a wide, flat band
    """
    # additional parameters to generate the flat band
    wC = 10
    v = 10
    gamma = 1

    beta = 1.0 / T
    Cut  = np.pi / dt

    t    = np.arange(0, tmax, dt)
    w    = np.arange(-Cut, Cut, dw)

    Delta = np.zeros((2, 2, len(t), len(t)), complex)  # indices are gtr/les | spin up/spin down \ times

    # frequency-domain Hybridization function
    def Hyb_gtr(w, t1, t2):
        return (1 / np.pi) * flatBand(w, wC, v, gamma) * (1 - fermi_function(w, beta, tune_mu(t2, mu))) * np.exp(-1j * w * (t1-t2))

    def Hyb_les(w, t1, t2):
        return (1 / np.pi) * flatBand(w, wC, v, gamma) * fermi_function(w, beta, tune_mu(t2, mu)) * np.exp(-1j * w * (t1-t2))

    def Integrate_Hyb(Hyb, t1, t2):
        return (quad(lambda w: np.real(Hyb(w, t1, t2)), -np.inf, np.inf, limit=300)[0] +
                1j * quad(lambda w: np.imag(Hyb(w, t1, t2)), -np.inf, np.inf, limit=300)[0])

    for t1 in range(len(t)):
        for t2 in range(len(t)):
            Delta[0,:,t2,t1] = np.exp(-1j*quad(lambda t: tune_mu(t,mu), t[t2], t[t1])[0]) * Integrate_Hyb(Hyb_gtr,t[t1],t[t2])
            Delta[1,:,t1,t2] = np.exp(-1j*quad(lambda t: tune_mu(t,mu), t[t2], t[t1])[0]) * Integrate_Hyb(Hyb_les,t[t1],t[t2])

            # Delta[0,:,t1,t2] = Integrate_Hyb(Hyb_gtr,t[t1],t[t2])
            # Delta[1,:,t2,t1] = Integrate_Hyb(Hyb_les,t[t1],t[t2])

    np.savez_compressed('Delta', t=t, D=Delta)
