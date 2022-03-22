import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.integrate import quad

w_0 = 1
delta_width = 0.1
v = 10
rho = 6

def tdiff(D, t2, t1):
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def coth(w, beta):
    return (np.exp(w*beta/2)+1.0)/(np.exp(w*beta/2)-1.0)

def PhononSpectral(w, mu):
    return (w-mu)/(rho-mu) / (np.exp(v*(w-rho)) + 1) # Ohmic bath
    # return (1 / w_0) * (np.exp(-((w-w_0)/delta_width)**2) / (np.sqrt(np.pi)*delta_width))  # Delta function
    # return (1/(np.pi*0.1)) * (1 / (1 + ((w-1)/0.1)**2))   # Lorenz distribution

def PhononCorrelation(w, t, beta):
    return coth(w,beta) * np.cos(w*t) - 1j * np.sin(w*t)
    # return 1/(np.exp(w*beta)-1)

def PhononBath(t, w, mu, beta):
    return PhononSpectral(w, mu) * w * PhononCorrelation(w, t, beta)

def genPhononBath(t, mu, T):
    phononBath_ = np.zeros((len(t)), complex)
    phononBath = np.zeros((len(t),len(t)), complex)
    beta = 1/T

    for t_ in range(len(t)):
        phononBath_[t_] = (quad(lambda w: np.real(PhononBath(t[t_], w, mu, beta)), 0, 10, limit=300)[0] + 1j * quad(lambda w: np.imag(PhononBath(t[t_], w, mu, beta)), 0, 10, limit=300)[0])

    for t1 in range(len(t)):
        for t2 in range(len(t)):
             phononBath[t1, t2] = tdiff(phononBath_, t1, t2)

    np.savez_compressed('PhononBath', t=t, P=phononBath)

    # w = np.arange(0, 10, 0.01)
    # plt.plot(w,PhononSpectral(w, mu),'r')
    # # plt.plot(w,PhononCorrelation(w),'r')
    # # plt.plot(w,PhononSpectral(w)*PhononCorrelation(w),'r')
    # # plt.plot(t, np.real(phononBath), 'r', t, np.imag(phononBath), 'b')
    # plt.grid()
    # plt.show()
