#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import argparse
import toml
import zipfile
import numpy as np
import os
import constant_electric_field

def calculate_current(U, F, mu1, mu2, T, dt, v_0, lattice_structure):
    # load Greens functions
    Green = 'Green_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(Green.format(U, F, mu1, mu2, T, dt))
    t = loaded['t']
    dt = t[1] - t[0]
    Green = loaded['Green']

    Delta_left = np.zeros((2, 2, len(t), len(t)), complex) # dynamical mean field
    Delta_right= np.zeros((2, 2, len(t), len(t)), complex) # dynamical mean field
    I = np.zeros((2, 2, len(t)), complex) # lattice current; spin up/spin down | time

    # v = electric_field.genv(pumpA, pumpOmega, params['t_pump_start'], params['t_pump_end'], probeA, probeOmega, params['t_probe_start'], params['t_probe_end'], params['v_0'], t, params['lattice_structure'])
    v = constant_electric_field.genv(F, v_0, t, lattice_structure)

    loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(mu1, T, dt))
    Delta_left = loaded['D'] / 2.0
    loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(mu2, T, dt))
    Delta_right = loaded['D'] / 2.0

    t_start = len(t)-100    
    t_cut = len(t)-1

    # # plt.plot(t[:t_cut], np.real(Green[0, 1, :t_cut, t_cut]),t[:t_cut], np.imag(Green[0, 1, :t_cut, t_cut])[:t_cut], '--', label='Green_T={}'.format(T))
    # # plt.plot(t[:t_cut], np.real(Delta_left[0, 1, :t_cut, t_cut]), t[:t_cut], np.imag(Delta_left[0, 1, :t_cut, t_cut]), '--', label='Delta_T={}'.format(T))
    # plt.plot(t[:t_cut], dt * np.real(Delta_left[0, 1, :t_cut, t_cut] * Green[0, 1, :t_cut, t_cut]), t[:t_cut], dt * np.imag(Delta_left[0, 1, :t_cut, t_cut] * Green[0, 1, :t_cut, t_cut]), '--', label='dt*Delta*Green_T={}'.format(T))
    # plt.legend()
    # plt.savefig('Greens_U={}_mu={}_dt={}.pdf'.format(U, mu, dt))

    # plt.plot(t[t_start:t_cut], np.real(Green[0, 1, t_start:t_cut, t_cut]), 'o', t[t_start:t_cut], np.imag(Green[0, 1, t_start:t_cut, t_cut]), '--', label='Green_dt={}'.format(dt))
    # plt.plot(t[t_start:t_cut], np.real(Delta_left[0, 1, t_start:t_cut, t_cut]), t[t_start:t_cut], np.imag(Delta_left[0, 1, t_start:t_cut, t_cut]), '--', label='Delta_dt={}'.format(dt))
    # plt.plot(t[t_start:t_cut], np.real(Delta_left[0, 1, t_cut, t_start:t_cut]), t[t_start:t_cut], np.imag(Delta_left[0, 1, t_cut, t_start:t_cut]), '--', label='Delta_dt={}'.format(dt))
    # plt.plot(t[:t_cut], dt * np.real(Delta_left[0, 1, :t_cut, t_cut] * Green[0, 1, :t_cut, t_cut]), t[:t_cut], dt * np.imag(Delta_left[0, 1, :t_cut, t_cut] * Green[0, 1, :t_cut, t_cut]), '--', label='dt*Delta*Green_dt={}'.format(dt))
    # plt.legend()
    # plt.savefig('Greens_U={}_mu={}_T={}.pdf'.format(U, mu, T))

    # calculate particle current in the lattice
    for t1 in range(len(t)):
        # left current
        if mu1 >= 0.0:
            # spin up 
            # I[0, 0, t1] = dt * np.trapz(Delta_left[0, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1])
            I[0, 0, t1] = dt * np.trapz(Delta_left[1, 1, :t1, t1-1] * (-Green[0, 1, :t1, t1-1] - Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta_left[1, 1, :t1, t1-1] + Delta_left[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
            # spin down
            # I[1, 0, t1] = dt * np.trapz(Delta_left[0, 0, :t1, t1-1] * Green[0, 0, :t1, t1-1])
            I[1, 0, t1] = dt * np.trapz(Delta_left[1, 0, :t1, t1-1] * (-Green[0, 0, :t1, t1-1] - Green[1, 0, :t1, t1-1])) + dt * np.trapz((Delta_left[1, 0, :t1, t1-1] + Delta_left[0, 0, :t1, t1-1]) * Green[1, 0, :t1, t1-1])
        else:
            # spin up 
            # I[0, 0, t1] = dt * np.trapz(Delta_left[1, 1, t1-1, :t1] * Green[1, 1, t1-1, :t1])
            I[0, 0, t1] = dt * np.trapz(Delta_left[1, 1, :t1, t1-1] * (-Green[0, 1, :t1, t1-1] - Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta_left[1, 1, :t1, t1-1] + Delta_left[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
            # spin down                                                                    
            # I[1, 0, t1] = dt * np.trapz(Delta_left[1, 0, t1-1, :t1] * Green[1, 0, t1-1, :t1])
            I[1, 0, t1] = dt * np.trapz(Delta_left[1, 0, :t1, t1-1] * (-Green[0, 0, :t1, t1-1] - Green[1, 0, :t1, t1-1])) + dt * np.trapz((Delta_left[1, 0, :t1, t1-1] + Delta_left[0, 0, :t1, t1-1]) * Green[1, 0, :t1, t1-1])

        if mu2 >= 0.0:
            # spin up 
            # I[0, 1, t1] = dt * np.trapz(Delta_right[0, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1])
            I[0, 1, t1] = dt * np.trapz(Delta_right[1, 1, :t1, t1-1] * (-Green[0, 1, :t1, t1-1] - Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta_right[1, 1, :t1, t1-1] + Delta_right[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
            # spin down
            # I[1, 1, t1] = dt * np.trapz(Delta_right[0, 0, :t1, t1-1] * Green[0, 0, :t1, t1-1])
            I[1, 1, t1] = dt * np.trapz(Delta_right[1, 0, :t1, t1-1] * (-Green[0, 0, :t1, t1-1] - Green[1, 0, :t1, t1-1])) + dt * np.trapz((Delta_right[1, 0, :t1, t1-1] + Delta_right[0, 0, :t1, t1-1]) * Green[1, 0, :t1, t1-1])
        else:
            # spin up 
            # I[0, 1, t1] = dt * np.trapz(Delta_right[1, 1, t1-1, :t1] * Green[1, 1, t1-1, :t1])
            I[0, 1, t1] = dt * np.trapz(Delta_right[1, 1, :t1, t1-1] * (-Green[0, 1, :t1, t1-1] - Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta_right[1, 1, :t1, t1-1] + Delta_right[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
            # spin down                                                                     
            # I[1, 1, t1] = dt * np.trapz(Delta_right[1, 0, t1-1, :t1] * Green[1, 0, t1-1, :t1])
            I[1, 1, t1] = dt * np.trapz(Delta_right[1, 0, :t1, t1-1] * (-Green[0, 0, :t1, t1-1] - Green[1, 0, :t1, t1-1])) + dt * np.trapz((Delta_right[1, 0, :t1, t1-1] + Delta_right[0, 0, :t1, t1-1]) * Green[1, 0, :t1, t1-1])

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
    # Ts = [1.0, 0.5]
    F = params['pumpA']
    mu1 = params['mu1']
    mu2 = params['mu2']
    # dts = [0.0125, 0.01, 0.0075, 0.005]
    dts = [0.01]
    # dt = 0.005
    tmax = params['tmax']
    pumpOmega = params['pumpOmega']

    # F = 0.0
    # Us = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    # Ts = [0.25, 0.5, 1.0]
    # Ts = [1.0]
    # mus = [0.0, 1.0, 2.0, 3.0]
    # mus = [0.0]

    for dt in dts:
    # for T in Ts: 
    #         for mu in mus:
        t, I = calculate_current(U, F, mu1, mu2, T, dt, params['v_0'], params['lattice_structure']) 
        I_left = I[0] 
        I_right = I[1]
        # plt.plot(t, np.real(I_left), label = 'I_left_dt = {}'.format(dt)) 
        # plt.plot(t, np.real(I_right), label = 'I_right_dt = {}'.format(dt)) 
        # plt.plot(t, np.real(I_right + I_left), label = 'I_right + I_left') 
        # plt.plot(t, np.real(I[0]), t, np.imag(I[0]), '--', label = 'I_left') 
        # plt.plot(t, np.real(I[1]), t, np.imag(I[1]), '--', label = 'I_right') 

        plt.plot(t, np.real(I[0, 0]), label = 'up_I_left_dt={}'.format(dt)) 
        plt.plot(t, np.real(I[0, 1]), label = 'up_I_right_dt={}'.format(dt)) 
        plt.plot(t, np.real(I[1, 0]), label = 'down_I_left_dt={}'.format(dt)) 
        plt.plot(t, np.real(I[1, 1]), label = 'down_I_right_dt={}'.format(dt)) 

        # plt.plot(t, np.imag(I[0, 0]), '--', label = 'up_I_left_dt={}'.format(dt)) 
        # plt.plot(t, np.imag(I[0, 1]), '--', label = 'up_I_right_dt={}'.format(dt)) 
        # plt.plot(t, np.imag(I[1, 0]), '--', label = 'down_I_left_dt={}'.format(dt)) 
        # plt.plot(t, np.imag(I[1, 1]), '--', label = 'down_I_right_dt={}'.format(dt)) 

        # plt.axhline(y=0.0, color='black', linestyle='-')    
        # plt.savefig('impurity_current_mu1={}_mu2={}_U={}.pdf'.format(mu1, mu2, params['U']))

    plt.legend()
    plt.grid()
    plt.savefig('current_mu1={}_mu2={}_U={}_T={}.pdf'.format(mu1,mu2,U,T))

if __name__ == "__main__":
    main()
