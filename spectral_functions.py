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

def analyze_spectral(U, F, mu, T, tmax, dt):
    # load Greens functions
    Green = 'Green_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(Green.format(U, F, mu, -mu, T, dt))
    Green = loaded['Green']
    t_  = np.arange(0, tmax, dt)

    spin = 1 

    # fft parameters
    dt = dt
    Cut_w = 2 * np.pi / dt
    dw = dt
    Cut_t = 2 * np.pi / dw
    
    ft = np.arange(-Cut_t/2, Cut_t/2, dt)
    fw = np.arange(-Cut_w/2, Cut_w/2, dw)
    
    N_w = len(fw)
    N_t = len(ft)
    
    # convert to plotting time and frequency domain
    wmin = -10.0
    wmax = -wmin
    w = np.arange(wmin, wmax, dw)
    w_start = int((N_w/2 + int(wmin/dw)))
    w_end = int((N_w/2 + int(wmax/dw)))
    
    tmin = -tmax
    t = np.arange(tmin, tmax, dt)
    t_start = int((N_t/2 + int(tmin/dt)))
    t_end = int((N_t/2 + int(tmax/dt)))
    
    # period = 2*np.pi/pumpOmega
    period = dt
    t_period = np.arange(tmax - period, tmax, dt)
    G_p = np.zeros((len(t_period), 4, len(t)), complex)           # second index: Gles | Ggtr | Gret | Gadv
    
    Heaviside_ret = np.zeros(len(t))
    Heaviside_adv = np.zeros(len(t))
    for t_i in range(len(t)):
        Heaviside_ret[t_i] = heaviside(t[t_i])
        Heaviside_adv[-t_i] = heaviside(t[t_i])
        Heaviside_adv[0] = heaviside(-t[0])
    
    # period averaged Greens functions
    steps = 2
    for i in range(0, len(t_period), steps):
        t_p = np.arange(0, t_period[i], dt)
    
        t_0 = int(len(t_))
        t_lower = t_0 - int(len(t_p)) 
        t_upper = t_0 + int(len(t_p))
    
        G_p[i, 0, t_lower:t_0] = 1j * (Green[1, spin, 0:len(t_p), len(t_p)-1]) 
        G_p[i, 0, t_0:t_upper] = 1j * (Green[1, spin, len(t_p)-1,  0:len(t_p)][::-1])
    
        G_p[i, 1, t_lower:t_0] = 1j * (Green[0, spin, len(t_p)-1, 0:len(t_p)])
        G_p[i, 1, t_0:t_upper] = 1j * (Green[0, spin, 0:len(t_p), len(t_p)-1][::-1])
    
        # Gret = -Theta(t)*(Ggtr - Gles)
        G_p[i, 2] = -Heaviside_ret*(G_p[i,1] + G_p[i,0])
    
        # Gadv = -Theta(-t)*(Ggtr - Gles)
        G_p[i, 3] = Heaviside_adv*(G_p[i,1] + G_p[i,0])
    
    G = np.sum(G_p,axis=0) / len(t_period)
    
    plt.plot(t, np.real(G[0]), label = 'G_0_real')
    plt.plot(t, np.imag(G[0]), label = 'G_0_imag')
    plt.legend()
    plt.grid()
    plt.savefig('greens_functions_spin={}.pdf'.format(spin))
    plt.close()
    
    # # save energy current as .out file
    # np.savetxt(path +'/Gles.out', G[0].view(float))
    # np.savetxt(path +'/Ggtr.out', G[1].view(float))
    # np.savetxt(path +'/Gret.out', G[2].view(float))
    # np.savetxt(path +'/Gadv.out', G[3].view(float))
    
    #######################################################################################################################################
    # fft of Greens functions
    G_N = np.zeros((4, N_t), complex) #  Gles | Ggtr | Gret | Gadv
    G_N[:, t_start:t_end] = G
    
    fG = ifftshift(fft(fftshift(G_N))) * dt / np.pi
    A = (fG[1] + fG[0]) 
    A_les = fG[0]
    A_gtr = fG[1]
    # f = np.imag(fG[0]) / (np.imag(A))
    f = np.imag(fG[0]) / (np.imag(fG[0]) + np.imag(fG[1]))

    I = np.imag(A_les[w_start:w_end]) - fermi_function(w, 0, 1.0/T) * np.imag(A[w_start:w_end])        
    # plt.plot(fw[w_start:w_end], np.imag(A[w_start:w_end]), label = 'G_ret+G_adv')
    # plt.plot(fw[w_start:w_end], np.imag(fG[0,w_start:w_end] + fG[1,w_start:w_end]), label = 'G_les+G_gtr')
    plt.plot(fw[w_start:w_end], np.imag(A_les[w_start:w_end]), label = 'G_les_dt={}'.format(dt))
    plt.plot(fw[w_start:w_end], np.imag(A_gtr[w_start:w_end]), label = 'G_gtr_dt={}'.format(dt))
    # plt.plot(fw[w_start:w_end], fermi_function(w, 0, 1.0/T) * np.imag(A[w_start:w_end]), label = '-fermi*(G_ret-G_a)_dt={}'.format(dt))
    # plt.plot(fw[w_start:w_end], -f[w_start:w_end] * np.imag(A[w_start:w_end]), label = '-fermi*(G_ret-G_a)_dt={}'.format(dt))
    # plt.plot(fw[w_start:w_end], I, label = 'I_dt={}'.format(dt))
    # plt.plot(fw[w_start:w_end], f[w_start:w_end], label = 'dt = {}'.format(dt))
    # plt.plot(w, fermi_function(w, 0, 1.0/T), color='black')

    plt.legend()
    plt.grid()
    plt.savefig('fft_greens_functions_spin={}.pdf'.format(spin))

def main():
    parser = argparse.ArgumentParser(description = "calculate couble occupation")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    
    U = params['U']
    T = params['T']
    F = params['pumpA']
    mu = params['mu']
    dt = params['dt'] 
    tmax = params['tmax']
    
    analyze_spectral(U, F, params['mu'], params['T'], params['tmax'], params['dt'])

if __name__ == "__main__":
    main()
