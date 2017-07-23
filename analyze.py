#!/usr/bin/env python3

import numpy as np
import argparse
import h5py
import hdf5

import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, ifftshift

def plot_spectrum(fname):
    with h5py.File(fname, "r") as h5f:
        tmax = h5f["dmft/params/tmax"].value
        dt   = h5f["dmft/params/dt"].value
        dw   = h5f["dmft/params/dw"].value

        t = np.arange(0, tmax, dt)
        w = np.arange(-8, 8, dw)

        iteration = -1 # load last iteration
        data_path = sorted([x[1].name for x in h5f["dmft/iterations"].items()])[iteration]
        print("loading data from {:s}".format(data_path))

        Ggtr = -1.0j * hdf5.h5_load(h5f, data_path + "/gf/gtr_up/data")[-1, ::-1]
        Gles =  1.0j * hdf5.h5_load(h5f, data_path + "/gf/les_up/data")[-1, ::-1]

        Cut = np.pi / dt

        N = int(Cut/dt)
        Gadv = np.zeros(N+1, complex)
        Gadv[0:int(len(t))] = (Gles - Ggtr)

        fGadv = fftshift(fft(Gadv)) / (np.pi)

        a = int((N-len(w))/2)
        b = int((N+len(w))/2)

        plt.plot(w, np.imag(fGadv[a:b]), 'b--', w, np.real(fGadv[a:b]), 'r--')
        plt.ylabel('A($\omega$)')
        plt.xlabel('$\omega$')
        plt.grid()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description = "analyze dmft")
    parser.add_argument("file", help="file to analyze")
    args = parser.parse_args()
    plot_spectrum(args.file)

if __name__ == "__main__":
    main()
