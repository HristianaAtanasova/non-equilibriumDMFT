#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import argparse
import toml
import zipfile
import numpy as np
import os
import constant_electric_field

def calculate_current(U, F, T, v_0, lattice_structure):
    # load Greens functions
    Green = 'Green_U={}_F={}_T={}.npz'
    loaded = np.load(Green.format(U, F, T))
    t = loaded['t']
    dt = t[1] - t[0]
    Green = loaded['Green']

    Delta = np.zeros((2, 2, len(t), len(t)), complex) # dynamical mean field
    I = np.zeros((2, len(t)), complex) # lattice current; spin up/spin down | time

    # v = electric_field.genv(pumpA, pumpOmega, params['t_pump_start'], params['t_pump_end'], probeA, probeOmega, params['t_probe_start'], params['t_probe_end'], params['v_0'], t, params['lattice_structure'])
    v = constant_electric_field.genv(F, v_0, t, lattice_structure)

    Delta[:, 0] = v * Green[:, 1]
    Delta[:, 1] = v * Green[:, 0]

    spin = 0 
    for t1 in range(len(t)):
        # calculate particle current in the lattice
        I[0, t1] = dt * np.trapz(Delta[1, 0, t1, :t1] * Green[1, 0, t1, :t1])
        I[1, t1] = dt * np.trapz(Delta[1, 1, t1, :t1] * Green[1, 1, t1, :t1])
        # I[0, t1] = dt * np.trapz(Delta[1, spin, :t1, t1] * Green[1, spin, :t1, t1])
        # I[1, t1] = dt * np.trapz(Delta[0, spin, :t1, t1] * Green[0, spin, :t1, t1])

    # I[0] = Green[1, 0].diagonal()
    # I[1] = Green[0, 1].diagonal()

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
    
    # Fs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    Fs = [0.0]
    for F in Fs:
        t, I = calculate_current(params['U'], F, params['T'], params['v_0'], params['lattice_structure']) 
        # plt.plot(t, np.real(I[0] + I[1]), label = 'F = {}'.format(F)) 
        plt.plot(t, np.real(I[0]), label = 'F = {}'.format(F)) 
        plt.plot(t, np.real(I[1]), '--') 
        plt.legend()
    
    plt.savefig('current_U={}.pdf'.format(params['U']))

if __name__ == "__main__":
    main()
