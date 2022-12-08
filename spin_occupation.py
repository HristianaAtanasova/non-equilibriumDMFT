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

def calculate_occupation(U, pumpA, mu1, mu2, T, dt):
    # load Greens functions
    K = 'K_1_f_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(K.format(U, pumpA, mu1, mu2, T, dt))
    t = loaded['t']
    K = loaded['K']

    return t, K[0].diagonal(), K[1].diagonal(), K[2].diagonal(), K[3].diagonal()

def main():
    parser = argparse.ArgumentParser(description = "calculate current")
    parser.add_argument("--params",   default = "run.toml")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    
    U = 5.0
    # F = 8.0
    # Us = [0.0, 2.0, 4.0, 6.0, 8.0]
    # Fs = [0.0, 2.0, 8.0, 12.0]
    Fs = [0.0]
    for F in Fs:
    # for U in Us:
        t, K0, K1, K2, K3 = calculate_occupation(U, F, params['mu'], params['mu'], params['T'], params['dt'])
        plt.plot(t, np.real(K2 + K3), label = 'up_F = {}'.format(F)) 
        plt.plot(t, np.real(K1 + K3), label = 'down_F = {}'.format(F)) 
        # plt.plot(t, np.real(K1 + K3), label = 'U = {}'.format(U)) 
        plt.axhline(y=0.5, color='black', linestyle='-')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('$n_{down}$')
    plt.savefig('spin_down_population_U={}.pdf'.format(U))
    # plt.savefig('spin_down_population_F={}.pdf'.format(F))

if __name__ == "__main__":
    main()
