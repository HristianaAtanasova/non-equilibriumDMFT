#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import scipy.optimize as opt
import scipy.special as special
import argparse
import toml
import zipfile
import numpy as np
import os
import matplotlib.pyplot as plt
# from sympy import Heaviside

def fermi_function(w, mu, beta):
    return 1 / (1 + np.exp(beta * (w - mu)))

def heaviside(x):
    if x == 0:
        return 0.5
    return 0 if x < 0 else 1

def calculate_dos(U, F, mu, T, tmax, dt, spin):
    # load Greens functions
    Green_path = 'Green_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(Green_path.format(U, F, mu, -mu, T, dt))
    t = loaded['t']
    dt = t[1] - t[0]
    Green = loaded['Green']

    # fft parameters
    dw = dt
    Cut_w = np.pi / dt
    dw = dt
    Cut_t = np.pi / dw
    
    ft = np.arange(-Cut_t, Cut_t, dt)
    fw = np.arange(-Cut_w, Cut_w, dw)
    
    N_w = len(fw)
    N_t = len(ft)
    
    # convert to plotting time and frequency domain
    wmin = -10.0
    wmax = 10.0
    w = np.arange(wmin, wmax, dw)
    w_start = int((N_w/2 + int(wmin/dw)))
    w_end = int((N_w/2 + int(wmax/dw)))
    
    tmin = -tmax
    t_double = np.arange(tmin, tmax, dt)
    t_start = int((N_t/2 + int(tmin/dt)))
    t_end = int((N_t/2 + int(tmax/dt)))
    
    G = np.zeros((4, len(t_double)), complex)  # first second index: Gles | Ggtr | Gret | Gadv
    Heaviside_ret = np.zeros(len(t_double))
    Heaviside_adv = np.zeros(len(t_double))
    for t1 in range(len(t_double)):
        Heaviside_ret[t1] = heaviside(t_double[t1])
        Heaviside_adv[-t1] = heaviside(t_double[t1])
        Heaviside_adv[0] = heaviside(-t_double[0])
    
    t_lower = 0 
    t_0 = int(len(t_double) / 2)
    t_upper = len(t_double)
    
    G[0, t_lower:t_0] = 1j * Green[1, spin, 0:len(t), len(t)-1]
    G[0, t_0:t_upper] = 1j * Green[1, spin, len(t)-1,  0:len(t)][::-1]
    
    G[1, t_lower:t_0] = 1j * Green[0, spin, len(t)-1, 0:len(t)]
    G[1, t_0:t_upper] = 1j * Green[0, spin, 0:len(t), len(t)-1][::-1]

    # Gret = -Theta(t)*(Ggtr - Gles)
    G[2] = Heaviside_ret * (G[1] + G[0])
    
    # Gadv = -Theta(-t)*(Ggtr - Gles)
    G[3] = Heaviside_adv * (G[1] + G[0])
    
    # fft of Greens functions
    G_N = np.zeros((4, N_t), complex) #  Gles | Ggtr | Gret | Gadv
    G_N[:, t_start:t_end] = G
    
    fG = ifftshift(fft(fftshift(G_N))) * dt/(2*np.pi)
    A = (fG[2]+fG[3]) / 2
    f = np.imag(fG[0]) / (2*np.imag(A))
    
    return fw[w_start:w_end], np.imag(A[w_start:w_end]), np.imag(fG[0,w_start:w_end])

def main():
    parser = argparse.ArgumentParser(description = "calculate couble occupation")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    
    spin = 0 
    w, dos, occupation = calculate_dos(params['U'], params['pumpA'], params['mu'], params['T'], params['tmax'], params['dt'], spin)

    plt.plot(w, dos, label = 'dos_spin = {}'.format(spin))
    plt.plot(w, occupation, label = 'N_spin = {}'.format(spin))
    plt.legend()
    plt.xlabel('$\omega$')
    plt.ylabel('dos($\omega$)')
    plt.savefig('dos_U={}.pdf'.format(params['U']))

if __name__ == "__main__":
    main()

