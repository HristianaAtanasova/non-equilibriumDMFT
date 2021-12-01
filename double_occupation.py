#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import argparse
import toml
import zipfile
import numpy as np
import os

def calculate_double_occupation(T, pumpA):
    Vertex = 'K_1_f_T={}.npz'
    loaded = np.load(Vertex.format(T))
    t = loaded['t']
    K = loaded['K']
    dt = t[1]-t[0]
    d = K[3].diagonal()

    plt.plot(t, np.real(d))
    plt.savefig('double_occupation_F={}.pdf'.format(pumpA))
    plt.close()

    return d.real

def main():
    parser = argparse.ArgumentParser(description = "calculate couble occupation")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))

    double_occ = calculate_double_occupation(params['T'], params['pumpA'])

if __name__ == "__main__":
    main()
