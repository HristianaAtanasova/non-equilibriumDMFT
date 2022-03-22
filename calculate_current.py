#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import argparse
import toml
import zipfile
import numpy as np
import os
import constant_electric_field

def heaviside(x):
    if x == 0:
        return 0.5
    return 0 if x < 0 else 1

def calculate_current(U, F, mu1, mu2, T, dt, v_0, lattice_structure):
    # load Greens functions
    Green = 'Green_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(Green.format(U, F, mu1, mu2, T, dt))
    t_ = loaded['t']
    dt = t_[1] - t_[0]
    tmax = len(t_) * dt
    Green = loaded['Green']

    tmin = -tmax
    t = np.arange(tmin, tmax, dt)
    t_cut = 0

    Delta = np.zeros((2, 2, len(t_), len(t_)), complex) # dynamical mean field
    I = np.zeros((2, len(t_)), complex) # lattice current; spin up/spin down | time
    G_p = np.zeros((4, len(t)), complex)           # second index: Gles | Ggtr | Gret | Gadv

    # v = electric_field.genv(pumpA, pumpOmega, params['t_pump_start'], params['t_pump_end'], probeA, probeOmega, params['t_probe_start'], params['t_probe_end'], params['v_0'], t, params['lattice_structure'])
    v = constant_electric_field.genv(F, v_0, t_, lattice_structure)

    Heaviside_ret = np.zeros(len(t))
    Heaviside_adv = np.zeros(len(t))
    for t_i in range(len(t)):
        Heaviside_ret[t_i] = heaviside(t[t_i])
        Heaviside_adv[-t_i] = heaviside(t[t_i])
        Heaviside_adv[0] = heaviside(-t[0])

    t_0 = int(len(t_))
    t_p = np.arange(0, tmax-t_cut, dt)
    t_lower = t_0 - int(len(t_p))
    t_upper = t_0 + int(len(t_p))

    G_p[0, t_lower:t_0] = 1j * (Green[1, 0, 0:len(t_p), len(t_p)-1] + Green[1, 1, 0:len(t_p), len(t_p)-1])
    G_p[0, t_0:t_upper] = 1j * (Green[1, 0, len(t_p)-1,  0:len(t_p)][::-1] + Green[1, 1, len(t_p)-1,  0:len(t_p)][::-1])
    G_p[1, t_lower:t_0] = 1j * (Green[0, 0, 0:len(t_p), len(t_p)-1] + Green[0, 1, 0:len(t_p), len(t_p)-1])
    G_p[1, t_0:t_upper] = 1j * (Green[0, 0, len(t_p)-1,  0:len(t_p)][::-1] + Green[0, 1, len(t_p)-1,  0:len(t_p)][::-1])

    G_p[0] = G_p[0] # Gles(t1,t2) 
    G_p[1] = -np.conj(G_p[1]) # Ggtr(t1,t2) 
    G_p[2] = -Heaviside_ret*(G_p[1] + G_p[0]) # Gret = -Theta(t)*(Ggtr - Gles)
    G_p[3] = Heaviside_adv*(G_p[1] + G_p[0]) # Gadv = -Theta(-t)*(Ggtr - Gles)
    
    # plt.plot(t, -np.real(G_p[2] - G_p[3]), t, -np.imag(G_p[2] - G_p[3]), '--', label='-G_ret-G_adv_dt={}'.format(dt))
    # plt.plot(t, np.real(G_p[0] + G_p[1]), t, np.imag(G_p[0] + G_p[1]), '--', label='G_les+G_gtr_dt={}'.format(dt))
    # plt.plot(t, np.real(G_p[0]), t, np.imag(G_p[0]), '--', label='Green_les_dt={}'.format(dt))
    # plt.plot(t, np.real(G_p[1]), t, np.imag(G_p[1]), '--', label='Green_gtr_dt={}'.format(dt))
    # plt.legend()
    # plt.show()
    # plt.savefig('Greens_U={}_T={}.pdf'.format(U, T))

    Delta[:, 1] = v * Green[:, 0]
    Delta[:, 0] = v * Green[:, 1]

    t = np.arange(0, tmax, dt)
    # t_start = len(t)-100    
    t_start = 0    
    t_cut = len(t) - 0
    # calculate particle current in the lattice
    for t1 in range(len(t)):
        # spin up 
        I[1, t1] = dt * np.trapz(Delta[1, 1, :t1, t1-1] * (-Green[0, 1, :t1, t1-1] - Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta[1, 1, :t1, t1-1] + Delta[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
        # spin down
        I[0, t1] = dt * np.trapz(Delta[1, 0, :t1, t1-1] * (-Green[0, 0, :t1, t1-1] - Green[1, 0, :t1, t1-1])) + dt * np.trapz((Delta[1, 0, :t1, t1-1] + Delta[0, 0, :t1, t1-1]) * Green[1, 0, :t1, t1-1])
    return t, I

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
    # Ts = [1.0, 0.5]
    F = params['pumpA']
    mu = params['mu']
    # dts = [0.0125, 0.01, 0.0075, 0.005]
    # dts = [0.01]
    dt = 0.01
    tmax = params['tmax']

    Fs = [0.0, 2.0, 4.0, 8.0]
    # Us = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    # Ts = [0.25, 0.5, 1.0]
    # Ts = [1.0]
    # mus = [0.0, 1.0, 2.0, 3.0]
    # mus = [0.0]

    for F in Fs:
    # for T in Ts: 
    #         for mu in mus:
        t, I = calculate_current(U, F, mu, mu, T, dt, params['v_0'], params['lattice_structure']) 
        plt.plot(t, np.real(I[0]), label = 'up_I_F={}'.format(F)) 
        # plt.plot(t, np.real(I[1]), label = 'down_I_F={}'.format(F)) 

    plt.legend()
    plt.grid()
    plt.savefig('current_U={}.pdf'.format(U, F))

if __name__ == "__main__":
    main()
