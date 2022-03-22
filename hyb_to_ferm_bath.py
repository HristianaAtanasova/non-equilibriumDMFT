import argparse
import toml
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import os
import zipfile

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def flatBand(w, wC, v, gamma):
    """
    DOS for a flat band with soft cutoff
    """
    return gamma / ((np.exp(v * (w - wC)) + 1) * (np.exp(-v * (w + wC)) + 1))

def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))

def genWideBandHyb(T, mu, t, dt, dw, wC, Lambda):
    """
    Generate energy hybridization function for Fermion bath with a wide, flat band
    """
    t  = np.arange(0, tmax, dt)
    # additional parameters to generate the flat band
    v = 10
    gamma = Lambda  # corresponds to Lambda!!!

    beta = 1.0 / T
    Cut  = np.pi / dt
    N = int(2*Cut/dw)
    w    = np.arange(-Cut, Cut, dw)

    Delta = np.zeros((2, 2, len(t), len(t)), complex)  # indices are gtr/les | spin up/spin down

    # frequency-domain Hybridization function
    Hyb_gtr = flatBand(w, wC, v, gamma) * (1 - fermi_function(w, beta, mu))
    Hyb_les = flatBand(w, wC, v, gamma) * fermi_function(w, beta, mu)

    fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/(np.pi)
    fDelta_les = ifftshift(fft(fftshift(Hyb_les))) * dw/(np.pi)

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

    return Delta

'''
Generate energy hybridization function
'''
cwd = os.getcwd()
Path = os.getcwd() + '/wC={}/U10.0_probeA0.0_probeOmega1.25_pumpA0.0_pumpOmega10.0/T{}'

tmax = 10
Temp = [1.0]
wC_ = [20, 30]

for T in Temp:
    for wC in wC_:
        path = Path.format(wC, T)
        parser = argparse.ArgumentParser(description = "run dmft")
        parser.add_argument("--params",   default = path + "/run.toml")
        args = parser.parse_args()

        with open(args.params, "r") as f:
            params = toml.load(f)

        params.update(vars(args))

        Delta = genWideBandHyb(T, params['mu'], params['tmax'], params['dt'], params['dw'], wC, params['Lambda'])

        Delta_name = 'ferm_Delta_wC={}_T={}_tmax{}'
        np.savez_compressed(Delta_name.format(wC,T,tmax), Delta=Delta)
