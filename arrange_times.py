#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import scipy.optimize as opt
import scipy.special as special
from numpy.linalg import inv
import argparse
import toml
import zipfile
import numpy as np
import os
import matplotlib.pyplot as plt


def heaviside(x):
    if x == 0:
        return 0.5
    return 0 if x < 0 else 1

def keldysh_to_real_time(Green, t, tmax):
    dt = t[1] - t[0]
    tmin = -tmax
    t_contour = np.arange(tmin, tmax, dt)
    G = np.zeros((2, 2, len(t), len(t)), complex)  # first second index: spin  

    t_max = int(len(t_contour) / 2)

    G21 = Green[:, t_max:len(t_contour), :t_max]
    G12 = Green[:, :t_max, t_max:len(t_contour)]
    for t1 in range(len(t)):
        G[1, :, t1, :(t1+1)] = G12[:, t1, -1:(-2-t1):-1]
        G[1, :, t1, t1:len(t)] = G12[:, t1, (-1-t1)::-1]
        G[0, :, t1, :t1] = G21[:, -1-t1, :t1]
        G[0, :, t1, t1:len(t)] = G21[:, -1-t1, t1:]

    G[1] = np.swapaxes(G[1], 1, 2)

    return G 

def real_time_to_keldysh(Green, t, tmax):
    dt = t[1] - t[0]
    tmin = -tmax
    t_contour = np.arange(tmin, tmax, dt)
    G_full_matrix = np.zeros((2, len(t_contour), len(t_contour)), complex)  # first second index: spin  
    G_contour = np.zeros((2, len(t_contour)), complex)  # first second index: spin  

    t_max = int(len(t_contour) / 2)

    G12 = np.zeros((2, len(t), len(t)), complex) 
    G21 = np.zeros((2, len(t), len(t)), complex) 
    G11 = np.zeros((2, len(t), len(t)), complex) 
    G22 = np.zeros((2, len(t), len(t)), complex) 

    Green[1] = np.swapaxes(Green[1], 1, 2)
    for t1 in range(len(t)):
        G12[0, t1, :t1] = Green[1, 0, t1, -1:(-1-t1):-1]
        G12[0, t1, t1:len(t)] = Green[1, 0, t1, (len(t)-1-t1)::-1]
        G21[0, t1, :t1] = Green[0, 0, (-1-t1), :t1]
        G21[0, t1, t1:len(t)] = Green[0, 0, (-1-t1), t1:len(t)]

        G12[1, t1, :(t1+1)] = Green[1, 1, t1, -1:(-2-t1):-1]
        G12[1, t1, t1:len(t)] = Green[1, 1, t1, (len(t)-1-t1)::-1]
        G21[1, t1, :(t1+1)] = Green[0, 1, (-1-t1), :(t1+1)]
        G21[1, t1, t1:len(t)] = Green[0, 1, (-1-t1), t1:len(t)]

    for t1 in range(len(t)):
        G11[0, t1, :t1] = Green[0, 0, t1, :t1] 
        G11[0, :(t1+1), t1] = Green[1, 0, :(t1+1), t1]
        G22[0, t1, :t1] = Green[0, 0, (-1-t1), -1:(-1-t1):-1]
        G22[0, :(t1+1), t1] = Green[1, 0, -1:(-2-t1):-1, (-1-t1)]

        G11[1, t1, :(t1+1)] = Green[0, 1, t1, :(t1+1)] 
        G11[1, :t1, t1] = Green[1, 1, :t1, t1]
        G22[1, t1, :(t1+1)] = Green[0, 1, (-1-t1), -1:(-2-t1):-1]
        G22[1, :t1, t1] = Green[1, 1, -1:(-1-t1):-1, (-1-t1)]

    G_full_matrix[:, t_max:(len(t_contour)+1), :(t_max)] = G21
    G_full_matrix[:, :t_max, t_max:len(t_contour)] = G12
    G_full_matrix[:, :t_max, :t_max] = G11
    G_full_matrix[:, t_max:len(t_contour), t_max:len(t_contour)] = G22
     
    Green[1] = np.swapaxes(Green[1], 1, 2)

    return G_full_matrix
    # return G11, G12, G21, G22, G_full_matrix

def calculate_ret_and_adv(Green, t, tmax):
    dt = t[1] - t[0]
    tmin = -tmax
    t_contour = np.arange(tmin, tmax, dt)
    G_full_matrix = np.zeros((2, len(t_contour), len(t_contour)), complex)  # first second index: spin  
    G_contour = np.zeros((2, len(t_contour)), complex)  # first second index: spin  

    t_max = int(len(t_contour) / 2)

    spin = 0
    G_contour[0, :t_max] = (Green[0, spin, len(t)-1, 0:len(t)]) 
    G_contour[0, t_max:len(t_contour)] = (Green[0, spin, 0:len(t), len(t)-1])[::-1] 
    G_contour[1, :t_max] = (Green[1, spin, len(t)-1, 0:len(t)])
    G_contour[1, t_max:len(t_contour)] = (Green[1, spin, 0:len(t), len(t)-1])[::-1]

    Heaviside_ret = np.zeros(len(t_contour))
    Heaviside_adv = np.zeros(len(t_contour))
    for t1 in range(len(t_contour)):
        Heaviside_ret[t1] = heaviside(t_contour[t1])
        Heaviside_adv[-t1] = heaviside(t_contour[t1])
        Heaviside_adv[0] = heaviside(-t_contour[0])

    G_ret = Heaviside_ret * (G_contour[1] + G_contour[0])
    G_adv = Heaviside_adv * (G_contour[1] + G_contour[0])

    # plt.plot(t_contour, np.real(G_contour[0]), '-', label = 'gtr_real')
    # plt.plot(t_contour, np.imag(G_contour[0]), '--', label = 'gtr_imag')
    # plt.plot(t_contour, np.real(G_contour[1]), '-', label = 'les_real')
    # plt.plot(t_contour, np.imag(G_contour[1]), '--', label = 'les_imag')
    # plt.legend()
    # plt.savefig('G_correct_gtr_les.pdf')
    # plt.close()

    # plt.plot(t_contour, np.real(G_ret), '-', label = 'ret_real')
    # plt.plot(t_contour, np.imag(G_ret), '--', label = 'ret_imag')
    # plt.plot(t_contour, np.real(G_adv), '-', label = 'adv_real')
    # plt.plot(t_contour, np.imag(G_adv), '--', label = 'adv_imag')
    # plt.legend()
    # plt.savefig('G_correct_ret_adv.pdf')
    # plt.close()

    G11, G12, G21, G22, G_full_matrix = real_time_to_keldysh(Green, t, tmax)

    # Gret = (G11 - G12 + G21 - G22) / 2.0
    # Gadv = (G11 + G12 - G21 - G22) / 2.0
    Gret = (G12 + G21) / 2.0
    Gadv = (G12 - G21) / 2.0

    plt.matshow(G_full_matrix[spin].real)
    plt.colorbar()
    plt.savefig('G_full_matrix_real.pdf')
    plt.close()

    plt.matshow(G_full_matrix[spin].imag)
    plt.colorbar()
    plt.savefig('G_full_matrix_imag.pdf')
    plt.close()

    G = np.zeros((4, len(t_contour)), complex) 

    G[0, :(t_max+1)] = G_full_matrix[spin, :(t_max+1), t_max] 
    G[0, t_max:] = G_full_matrix[spin, t_max, t_max:] 
    G[1, :(t_max+1)] = G_full_matrix[spin, t_max, :(t_max+1)] 
    G[1, t_max:] = G_full_matrix[spin, t_max:, t_max] 

    G[2, :t_max] = Gret[spin, :len(t), len(t)-1] 
    G[2, t_max:] = Gret[spin, len(t)-1, :len(t)] 
    G[3, :t_max] = Gadv[spin, len(t)-1, :len(t)] 
    G[3, t_max:] = Gadv[spin, :len(t), len(t)-1] 

    return G_full_matrix

def main():
    parser = argparse.ArgumentParser(description = "calculate couble occupation")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))

    Green_path = 'Green_U={}_F={}_mu1={}_mu2={}_T={}_dt={}.npz'
    loaded = np.load(Green_path.format(params['U'], params['pumpA'], params['mu'], -params['mu'], params['T'], params['dt']))
    Green = loaded['Green']
    # Green_path = 'Delta_mu={}_T={}_dt={}.npz'
    # loaded = np.load(Green_path.format(params['mu'], params['T'], params['dt']))
    # Green = loaded['D']

    t = loaded['t']
    dt = t[1] - t[0]
    
    # calculate_ret_and_adv(Green, t, params['tmax'])
    G_contour = real_time_to_keldysh(Green, t, params['tmax'])
    G_matrix = keldysh_to_real_time(G_contour, t, params['tmax'])

    # print(Green[0] == G_matrix[0])
    # print(Green[1] == G_matrix[1])

if __name__ == "__main__":
    main()

