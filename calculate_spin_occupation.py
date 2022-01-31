import argparse
import toml
import zipfile
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import os
# import electric_field
import constant_electric_field

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def calculate_occupation(U, pumpA, T, v_0, lattice_structure):
    # load Greens functions
    K = 'K_1_f_U={}_F={}_T={}.npz'
    loaded = np.load(Green.format(U, pumpA, T))
    t = loaded['t']
    dt = t[1] - t[0]
    K = loaded['K']

    return K.diagonal()

def main():
    parser = argparse.ArgumentParser(description = "calculate current")
    parser.add_argument("--params",   default = "run.toml")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    
    Fs = [0.0]
    for F in Fs:
        K = calculate_occupation(params['U'], F, params['T'], params['v_0'], params['lattice_structure'])
        plt.plot(t, np.real(K[0] + K[3]), label = 'spin_up_F = {}'.format(F)) 
        plt.plot(t, np.real(K[1] + K[3]), label = 'spin_down_F = {}'.format(F)) 

    plt.legend()
    plt.savefig('spin_population.pdf')

if __name__ == "__main__":
    main()
