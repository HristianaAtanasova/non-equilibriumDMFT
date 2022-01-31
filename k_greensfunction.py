from scipy.signal import fftconvolve
import numpy as np
from scipy.linalg import inv, pinv, pinvh
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import argparse 
import toml
import hybridization 
import arrange_times

def trapezConv(a, b, dt):
    return dt * (fftconvolve(a, b)[:len(a)] - 0.5 * a[:] * b[0] - 0.5 * a[0] * b[:])


def tdiff(D, t2, t1):
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def tdiff_2(D, t2, t1):
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))


def hypercubicDos(w, v):
    """
    DOS for the cubic lattice in the limit d -> inf 
    """
    return np.exp(-(w ** 2) / v**2) / (np.sqrt(np.pi) * v) 
 

def gen_k_dep_Green(T, v_0, U, mu, tmax, dt, wC, dw):
    """
    Generate Hybridization function for Fermion bath with a semicircular DOS
    """
    beta = 1.0 / T
    t = np.arange(0, tmax, dt)
    t_contour = np.arange(-tmax, tmax, dt)
    w = np.arange(-wC, wC, dw)
    
    G_k = np.zeros((2, 2, len(t), len(w)), complex)
    G_dot = np.zeros((2, 2, len(t)), complex)
    Green_k = np.zeros((2, 2, len(t), len(t), len(w)), complex)
    Green_dot = np.zeros((2, 2, len(t), len(t)), complex)
    contour_Green_k = np.zeros((2, len(t_contour), len(t_contour), len(w)), complex)
    contour_Green_k_inv = np.zeros((2, len(t_contour), len(t_contour), len(w)), complex)

    dos = hybridization.hypercubicDos(w, v_0)
    fermi = fermi_function(w, beta, mu)
    for t1 in range(len(t)):
        G_k[0, 0, t1] = np.exp(-1j * t[t1] * w) * (1 - fermi)
        G_k[0, 1, t1] = np.exp(-1j * t[t1] * w) * (1 - fermi)
        G_k[1, 0, t1] = np.exp(-1j * t[t1] * w) * fermi
        G_k[1, 1, t1] = np.exp(-1j * t[t1] * w) * fermi
    
        # the initial state of the dot is a spin up state 
        G_dot[1, 0, t1] = np.exp(-1j * t[t1] * -(U / 2.0))
        G_dot[0, 1, t1] = np.exp(-1j * t[t1] * -(U / 2.0))

    # G = dw * np.trapz(G_k, x = w, axis = -1)
    for t1 in range(len(t)):
        for t2 in range(len(t)):
            Green_dot[0, 0, t2, t1] = tdiff(G_dot[0, 0], t2, t1)
            Green_dot[0, 1, t2, t1] = tdiff(G_dot[0, 1], t2, t1)
            Green_dot[1, 0, t1, t2] = tdiff(G_dot[1, 0], t2, t1)
            Green_dot[1, 1, t1, t2] = tdiff(G_dot[1, 1], t2, t1)
            
            for w1 in range(len(w)):
                Green_k[0, 0, t2, t1, w1] = tdiff(G_k[0, 0, :, w1], t2, t1)
                Green_k[0, 1, t2, t1, w1] = tdiff(G_k[0, 1, :, w1], t2, t1)
                Green_k[1, 0, t1, t2, w1] = tdiff(G_k[1, 0, :, w1], t2, t1)
                Green_k[1, 1, t1, t2, w1] = tdiff(G_k[1, 1, :, w1], t2, t1)

    for w1 in range(len(w)):
        contour_Green_k[:,:,:,w1] = arrange_times.real_time_to_keldysh(Green_k[:,:,:,:,w1], t, tmax) 
    
    contour_Green_dot = arrange_times.real_time_to_keldysh(Green_dot, t, tmax) 
    
    for spin in range(2):
        for w1 in range(len(w)):
            contour_Green_k_inv[spin, :, :, w1] = inv(contour_Green_k[spin, :, :, w1]) 

    # plt.matshow(contour_Green_k[0,0].real)
    # plt.colorbar()
    # plt.show()

    # plt.matshow(contour_Green_k[0,0].imag)
    # plt.colorbar()
    # plt.show()
    
    np.savez_compressed('Green_k_inv', G_k=contour_Green_k,  G_k_inv=contour_Green_k_inv, G_d=contour_Green_dot)

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


