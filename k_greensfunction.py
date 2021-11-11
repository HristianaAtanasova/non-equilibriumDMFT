from scipy.signal import fftconvolve
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import argparse 
import toml


def trapezConv(a, b, dt):
    return dt * (fftconvolve(a, b)[:len(a)] - 0.5 * a[:] * b[0] - 0.5 * a[0] * b[:])


def tdiff(D, t1, t2):
    return D[:, :, t2 - t1] if t2 >= t1 else np.conj(D[:, :, t1 - t2])


def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))


def hypercubicDos(w, v):
    """
    DOS for the cubic lattice in the limit d -> inf 
    """
    return np.exp(-(w ** 2) / v**2) / (np.sqrt(np.pi) * v) 
 

def gen_k_dep_Green(T, v_0, mu, tmax, dt, wC, dw):
    """
    Generate Hybridization function for Fermion bath with a semicircular DOS
    """
    beta = 1.0 / T
    wC = 10 

    t = np.arange(0, tmax, dt)
    w = np.arange(-wC, wC, dw)
    
    G_k = np.zeros((2, 2, len(t), len(w)), complex)
    Green_k = np.zeros((2, 2, len(t), len(t), len(w)), complex)
    Green_k_inv = np.zeros((2, 2, len(t), len(t), len(w)), complex)

    Dos = hypercubicDos(w, v_0)
    Fermi = fermi_function(w, beta, mu)

    for t1 in range(len(t)):
        G_k[0, :, t1] = 1j * np.exp(-1j * t[t1] * w) * (1 - Fermi) * Dos
        G_k[1, :, t1] = 1j * np.exp(-1j * t[t1] * w) * Fermi * Dos
    
    for t1 in range(len(t)):
        for t2 in range(len(t)):
            Green_k[:, :, t1, t2] = tdiff(G_k, t1, t2)

    for w1 in range(len(w)):
        Green_k_inv[0, 0, :, :, w1] = inv(Green_k[0, 0, :, :, w1]) 
        # print('Green = ', Green_k[0, 0, :, :, w1])
        print(np.matmul(Green_k[0, 0, :, :, w1], Green_k_inv[0, 0, :, :, w1]))
        Green_k_inv[0, 1, :, :, w1] = inv(np.real(Green_k[0, 1, :, :, w1])) + 1j * inv(np.imag(Green_k[0, 1, :, :, w1]))  
        Green_k_inv[1, 0, :, :, w1] = inv(np.real(Green_k[1, 0, :, :, w1])) + 1j * inv(np.imag(Green_k[1, 0, :, :, w1]))
        Green_k_inv[1, 1, :, :, w1] = inv(np.real(Green_k[1, 1, :, :, w1])) + 1j * inv(np.imag(Green_k[1, 1, :, :, w1]))
    np.savez_compressed('Green_k_inv', D=Dos, G=Green_k_inv)

def main():
    parser = argparse.ArgumentParser(description = "run dmft")
    # parser.add_argument("--output",   default = "output.h5")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))

    Green = gen_k_dep_Green(params['T'], params['v_0'], params['mu'], params['tmax'], params['dt'], params['wC'], params['dw']) 

if __name__ == "__main__":
    main()


