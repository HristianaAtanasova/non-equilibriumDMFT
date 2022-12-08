#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import argparse
import toml
import zipfile
import numpy as np
import os
import constant_electric_field
import arrange_times 

def calculate_current(U, F, mu1, mu2, T, tmax, dt, v):
    # load Greens functions
    Green = 'Green_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(Green.format(U, F, mu1, mu2, T, dt))
    t = loaded['t']
    Green = loaded['Green']

    Delta_left = np.zeros((2, 2, len(t), len(t)), complex) # dynamical mean field
    Delta_right= np.zeros((2, 2, len(t), len(t)), complex) # dynamical mean field
    I = np.zeros((2, 2, len(t)), complex) # lattice current; spin up/spin down | time

    loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(mu1, T, dt))
    Delta_left = loaded['D'] / 2.0
    loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(mu2, T, dt))
    Delta_right = loaded['D'] / 2.0

    t_start = len(t)-100    
    t_cut = len(t)-1

    # calculate particle current in the lattice
    for t1 in range(len(t)):
        # left current
        # spin up 
        I[0, 0, t1] = - dt * np.trapz(Delta_left[1, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1]) + dt * np.trapz(Delta_left[0, 1, :t1, t1-1] * Green[1, 1, :t1, t1-1])
        # spin down
        I[1, 0, t1] = - dt * np.trapz(Delta_left[1, 0, :t1, t1-1] * Green[0, 0, :t1, t1-1]) + dt * np.trapz(Delta_left[0, 0, :t1, t1-1] * Green[1, 0, :t1, t1-1])
        # spin up 
        I[0, 1, t1] = - dt * np.trapz(Delta_right[1, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1]) + dt * np.trapz(Delta_right[0, 1, :t1, t1-1] * Green[1, 1, :t1, t1-1])
        # spin down
        I[1, 1, t1] = - dt * np.trapz(Delta_right[1, 0, :t1, t1-1] * Green[0, 0, :t1, t1-1]) + dt * np.trapz(Delta_right[0, 0, :t1, t1-1] * Green[1, 0, :t1, t1-1])
    return t, I

def main():
    parser = argparse.ArgumentParser(description = "calculate couble occupation")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run_impurity.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    
    U = params['U']
    T = params['T']
    F = params['pumpA']
    mu1 = params['mu1']
    mu2 = params['mu2']
    dt = params['dt']
    tmax = params['tmax']
    t = np.arange(0, tmax, dt)

    # v = electric_field.genv(F, params['pumpOmega'], params['t_pump_start'], params['t_pump_end'], params['probeA'], params['probeOmega'], params['t_probe_start'], params['t_probe_end'], params['v_0'], t, params['lattice_structure'])
    v = constant_electric_field.genv(F, params['v_0'], t, params['t_pump_start'], params['t_pump_end'], params['lattice_structure'])

    t, I = calculate_current(U, F, mu1, mu2, T, tmax, dt, v) 

    plt.plot(t, np.real(I[0, 0] + I[0, 1]), label = 'up') 
    plt.plot(t, np.real(I[1, 0] + I[1, 1]), label = 'down') 

    # plt.axhline(y=0.0, color='black', linestyle='-')    
    # plt.savefig('impurity_current_mu1={}_mu2={}_U={}.pdf'.format(mu1, mu2, params['U']))

    plt.legend()
    plt.grid()
    plt.savefig('impurity_current_mu1={}_mu2={}_U={}_T={}.pdf'.format(mu1,mu2,U,T))

if __name__ == "__main__":
    main()
