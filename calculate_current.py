#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import argparse
import toml
import zipfile
import numpy as np
import os
import constant_electric_field
import electric_field
import arrange_times

def heaviside(x):
    if x == 0:
        return 0.5
    return 0 if x < 0 else 1

def calculate_current(U, F, mu, T, dt, v):
    # load Greens functions
    Green = 'Green_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(Green.format(U, F, mu, -mu, T, dt))
    t_ = loaded['t']
    tmax = len(t_) * dt
    Green = loaded['Green']
    tmin = -tmax
    t = np.arange(tmin, tmax, dt)
    t_cut = 0

    Delta = np.zeros((2, 2, len(t_), len(t_)), complex) # dynamical mean field
    I = np.zeros((2, 2, len(t_)), complex)              # lattice current; spin up/spin down | time

    Delta[:, 1] = Green[:, 0] 
    Delta[:, 0] = Green[:, 1] 

    Delta_left = v * Delta / 2.0
    Delta_right = np.conj(v) * Delta / 2.0

    # loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(mu, T, dt))
    # Delta_left = v * loaded['D'] / 2.0
    # loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(-mu, T, dt))
    # Delta_right = np.conj(v) * loaded['D'] / 2.0
    # Delta = Delta_left + Delta_right

    t = np.arange(0, tmax, dt)
    # t_start = len(t)-100    
    t_start = 0    
    t_cut = len(t) - 0

    # calculate particle current in the lattice
    for t1 in range(len(t)):
        # left current
        # spin up 
        I[0, 0, t1] = - dt * np.trapz(Delta_left[1, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1]) + dt * np.trapz(Delta_left[0, 1, :t1, t1-1] * Green[1, 1, :t1, t1-1])
        # I[0, 0, t1] = dt * np.trapz(Delta_left[1, 1, :t1, t1-1] * (-Green[0, 1, :t1, t1-1] - Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta_left[1, 1, :t1, t1-1] + Delta_left[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
        # spin down
        I[1, 0, t1] = - dt * np.trapz(Delta_left[1, 0, :t1, t1-1] * Green[0, 0, :t1, t1-1]) + dt * np.trapz(Delta_left[0, 0, :t1, t1-1] * Green[1, 0, :t1, t1-1])
        # I[1, 0, t1] = dt * np.trapz(Delta_left[1, 0, :t1, t1-1] * (-Green[0, 0, :t1, t1-1] - Green[1, 0, :t1, t1-1])) + dt * np.trapz((Delta_left[1, 0, :t1, t1-1] + Delta_left[0, 0, :t1, t1-1]) * Green[1, 0, :t1, t1-1])
        # spin up 
        I[0, 1, t1] = - dt * np.trapz(Delta_right[1, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1]) + dt * np.trapz(Delta_right[0, 1, :t1, t1-1] * Green[1, 1, :t1, t1-1])
        # I[0, 1, t1] = dt * np.trapz(Delta_right[1, 1, :t1, t1-1] * (-Green[0, 1, :t1, t1-1] - Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta_right[1, 1, :t1, t1-1] + Delta_right[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
        # spin down
        I[1, 1, t1] = - dt * np.trapz(Delta_right[1, 0, :t1, t1-1] * Green[0, 0, :t1, t1-1]) + dt * np.trapz(Delta_right[0, 0, :t1, t1-1] * Green[1, 0, :t1, t1-1])
        # I[1, 1, t1] = dt * np.trapz(Delta_right[1, 0, :t1, t1-1] * (-Green[0, 0, :t1, t1-1] - Green[1, 0, :t1, t1-1])) + dt * np.trapz((Delta_right[1, 0, :t1, t1-1] + Delta_right[0, 0, :t1, t1-1]) * Green[1, 0, :t1, t1-1])
    return t, I

    # for t1 in range(len(t)):
    #    # spin up 
    #    # I[1, t1] = - dt * np.trapz(Delta[1, 1, :t1, t1-1] * (Green[0, 1, :t1, t1-1] + Green[1, 1, :t1, t1-1])) + dt * np.trapz((Delta[1, 1, :t1, t1-1] + Delta[0, 1, :t1, t1-1]) * Green[1, 1, :t1, t1-1])
    #    I[1,t1] = - dt * np.trapz(Delta[1, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1]) + dt * np.trapz(Delta[0, 1, :t1, t1-1] * Green[1, 1, :t1, t1-1])

    #    # spin down
    #    # I[0, t1] = - dt * np.trapz(Delta[1, 0, :t1, t1-1] * Green[0, 0, :t1, t1-1]) + dt * np.trapz(Delta[0, 0, :t1, t1-1] * Green[1, 0, :t1, t1-1])
    #    I[0, t1] = - dt * np.trapz(Delta[1, 1, :t1, t1-1] * Green[0, 1, :t1, t1-1]) + dt * np.trapz(Delta[0, 1, :t1, t1-1] * Green[1, 1, :t1, t1-1])

    # return t, I

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
    t = np.arange(0, tmax, dt)

    # v = electric_field.genv(F, params['pumpOmega'], params['t_pump_start'], params['t_pump_end'], params['probeA'], params['probeOmega'], params['t_probe_start'], params['t_probe_end'], params['v_0'], t, 0)
    v = constant_electric_field.genv(F, params['v_0'], t, params['t_pump_start'], params['t_pump_end'], 1)

    t, I = calculate_current(U, F, mu, T, dt, v) 

    # plt.plot(t, np.real(I[0, 0]), label = 'up_I_left_dt={}'.format(dt))
    # plt.plot(t, np.real(I[0, 1]), label = 'up_I_right_dt={}'.format(dt))
    # plt.plot(t, np.real(I[1, 0]), label = 'down_I_left_dt={}'.format(dt))
    # plt.plot(t, np.real(I[1, 1]), label = 'down_I_right_dt={}'.format(dt))

    plt.plot(t, np.real(I[0, 0] + I[0, 1]), label = 'up')
    plt.plot(t, np.real(I[1, 0] + I[1, 1]), label = 'down')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('I(t)')
    plt.grid()
    plt.savefig('current_mu={}_U={}_F={}_T={}.pdf'.format(mu, U, F, T))


if __name__ == "__main__":
    main()
