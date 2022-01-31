#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import argparse
import toml
import zipfile
import numpy as np
import os

def calculate_double_occupation(U, F, T):
    Vertex = 'K_1_f_U={}_F={}_T={}.npz'
    loaded = np.load(Vertex.format(U, F, T))
    t = loaded['t']
    K = loaded['K']
    dt = t[1]-t[0]
    d = K[2].diagonal() + K[3].diagonal()
    return t, d.real

def main():
    parser = argparse.ArgumentParser(description = "calculate couble occupation")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    
    Fs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    for F in Fs:
        t, d = calculate_double_occupation(params['U'], F, params['T']) 
        plt.plot(t, np.real(d), label='F = {}'.format(F))
        plt.legend()

    plt.savefig('spin_down_occupation_U={}.pdf'.format(params['U']))
    plt.close()

if __name__ == "__main__":
    main()
