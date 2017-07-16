from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime
from scipy.integrate import quad
from scipy.signal import hilbert
from numpy.linalg import inv
from numpy.linalg import solve

########################################################################################################################
''' Calculate Hybridization function for the Fermion Bath and the Phonon Bath '''

########## Define functions ###########

def coth(w):
    return (np.exp(w)+1.0)/(np.exp(w)-1.0)


def phononSpectral(w):
    # return (w-mu)/(rho-mu) / (np.exp(v*(w-rho)) + 1)
    return (np.pi/2) * np.exp(-((w-w_0)/delta_width)**2) / w_0


def phononCorrelation(t, w):
    return coth(beta*w/2) * np.cos(w*t) - 1j * np.sin(w*t)


def phononBath_(t,w):
    return (2/np.pi) * phononSpectral(w) * w * phononCorrelation(t,w)


# fermi distribution
def fermi_function(w):
    return 1 / (1 + np.exp(beta * (w - mu)))


# bose distribution
def bose_function(w):
    return 1 / (np.exp(beta * w) - 1)


# flat band with soft cutoff
def A(w):
    return 1 / ((np.exp(v * (w - wC)) + 1) * (np.exp(-v * (w + wC)) + 1))


# semicircular density of states for bethe lattice
def semicircularDos(w):
    return 1 / (2 * np.pi * v_0 ** 2) * np.sqrt(4 * v_0 ** 2 - w ** 2)


# fill in initial DeltaMatrix for two contour times with initial Delta for time-differences
def initMatrix(Delta_init, phononBath, phononCoupling):
    for t1 in range(len(t)):
        for t2 in range(len(t)):

            Delta[:, :, :, t1, t2] = tdiff(Delta_init[0, 0], t1, t2)

            PhononBath[t1, t2] = tdiff(phononBath, t1, t2)

            PhononCoupling[t1, t2] = phononCoupling[t1]*phononCoupling[t2]

########## Calculate Hybridization functions ###########
def generate(Cut, dw, v_0, t_, wDOS):

    # window function padded with zeros for semicircular DOS
    N = int(2*Cut/dw)
    a = int(N/2+2*v_0/dw)
    b = int(N/2-2*v_0/dw)
    DOS = np.zeros(N+1)
    DOS[b:a] = semicircularDos(wDOS)
    #DOS = hilbert(DOS)

    # frequency-domain Hybridization function
    Hyb_les = DOS * fermi_function(w)
    Hyb_gtr = DOS * (1 - fermi_function(w))

    # Hyb_les = A(w) * fermi_function(w)
    # Hyb_gtr = A(w) * (1 - fermi_function(w))


    fDelta_les = np.conj(ifftshift(fft(fftshift(Hyb_les)))) * dw/np.pi
    fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi


    # get real times from fft_times
    Delta_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down
    phononBath = np.zeros((len(t)), complex)

    for t_ in range(len(t)):

        Delta_init[0, :, t_] = fDelta_gtr[int(N / 2) + t_]
        Delta_init[1, :, t_] = fDelta_les[int(N / 2) + t_]

        phononBath[t_] = (2/np.pi) * (quad(lambda w: np.real(phononBath_(t[t_],w)), -100, -0.01, limit=600)[0] + quad(lambda w: np.real(phononBath_(t[t_],w)), 0.01, 100, limit=600)[0]
                                      + 1j * quad(lambda w: np.imag(phononBath_(t[t_],w)), -100, -0.01, limit=600)[0] + 1j * quad(lambda w: np.imag(phononBath_(t[t_],w)), 0.01, 100, limit=600)[0])

        # phononBath[t_] = (2 / np.pi) * (quad(lambda w: np.real(phononBath_(t[t_], w)), 0, 10, limit=300)[0] + 1j * quad(lambda w: np.imag(phononBath_(t[t_], w)), 0, 10, limit=300)[0])


    initMatrix(Delta_init, phononBath, phononCoupling)