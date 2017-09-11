#!/usr/bin/env python3

import numpy as np
from numpy.linalg import solve

import h5py
import hdf5

import matplotlib.pyplot as plt

from scipy.signal import fftconvolve
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime

########## Define functions for impurity solver ##########

def delta(f, i):
    return 1 if f == i else 0

def weights(A):
    if len(A) == 1:
        return 0
    if len(A) == 2:
        return 0.5*(A[0]+A[1])
    if len(A) == 3:
        return (A[0]+4*A[1]+A[2])/3
    return np.sum(A) - (A[0]+A[-1])*2/3 + (A[1]+A[-2])*7/24 - (A[2]+A[-3])*1/6 + (A[3]+A[-4])*1/24

def w(A):
    if len(A) == 1:
        return 0
    if len(A) == 2:
        return (1/2)**2
    if len(A) == 3:
        return (1/3)**2
    if len(A) == 4:
        return (3/8)**2
    return (1/3)**2

# Self energy for vertex functions
def SelfEnergy(K, DeltaMatrix, Sigma):
    for f in range(4):
        for j in range(4):
            Sigma[f] += K[j] * DeltaMatrix[j, f]

def fillDeltaMatrix(DeltaMatrix, Delta):
    # fill in DeltaMatrix for all times | first index is gtr/les | second is spin up/spin down

    # initial dot state |0> can only go to |up> or |down> with a Delta_gtr
    DeltaMatrix[0, 1] = Delta[1, 0]
    DeltaMatrix[0, 2] = Delta[1, 1]

    # initial dot state |up> can only go to |0> or |up,down>
    DeltaMatrix[1, 0] = Delta[0, 0]
    DeltaMatrix[1, 3] = Delta[1, 1]

    # initial dot state |down> can only go to |0> or |up,down>
    DeltaMatrix[2, 0] = Delta[0, 1]
    DeltaMatrix[2, 3] = Delta[1, 0]

    # initial dot state |up,down> can only go to |up> or |down> with a Delta_les
    DeltaMatrix[3, 1] = Delta[0, 1]
    DeltaMatrix[3, 2] = Delta[0, 0]

def bare_prop(t, U_):
    E = np.zeros((4, len(t)), float)

    E[1, :] = -U_/2.0
    E[2, :] = -U_/2.0

    # g_0 as a two-times Matrix for convolution integrals
    G_0 = np.zeros((4, len(t), len(t)), complex)

    # Computation of bare propagators G_0
    for i in range(4):
        for t1 in range(len(t)):
            for t2 in range(t1+1):
                G_0[i, t1, t2] = np.exp(-1j * E[i, t1-t2] * t[t1-t2])

    return G_0

def solve(t, U_, output, hybsection, gfsection):
    ########## Computation of bold propagators on separate branches of the contour ##########

    with h5py.File(output, "r") as h5f:
        Delta, _ = hdf5.load_green(h5f, hybsection)

    dt = t[1] - t[0]

    # initialize bare propagators
    start = datetime.now()

    # indices are initial and final states
    DeltaMatrix = np.zeros((4, 4, len(t), len(t)), complex)
    fillDeltaMatrix(DeltaMatrix, Delta)

    # indices are initial state, contour times located on the same branch t_n, t_m (propagation from t_m to t_n)
    G_0   = bare_prop(t, U_)
    G     = np.zeros((4, len(t), len(t)), complex)
    Sigma = np.zeros((4, len(t), len(t)), complex)

    # main loop over every pair of times t_n and t_m (located on the same contour-branch), where t_m is the smaller contour time
    # take integral over t1 outside the t_m loop
    for t_n in range(len(t)):
        sum_t1 = np.zeros((4, t_n+1), complex)
        for t_m in range(t_n, -1, -1):

            sum_t2 = np.zeros(4, complex)
            for i in range(4):
                sum_t2[i] = dt**2 * weights(G[i, t_m:t_n+1, t_m]*sum_t1[i, t_m:])
                # sum_t2[i] = dt ** 2 * np.trapz(G[i, t_m:t_n + 1, t_m] * sum_t1[i, t_m:])

            # Dyson equation for time (t_m, t_n)
            G[:, t_n, t_m] = (G_0[:, t_n, t_m] - sum_t2) / (1 + dt ** 2 * G_0[:, t_n, t_n] * Sigma[:, t_n, t_n] * w(G[i, t_m:t_n+1, t_m]*sum_t1[i, t_m:]))
            # G[:, t_n, t_m] = (G_0[:, t_n, t_m] - sum_t2) / (1 + dt ** 2 * G_0[:, t_n, t_n] * Sigma[:, t_n, t_n] * 1/4)

            # Compute self-energy for time (t_m, t_n)
            Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrix[:, :, t_n, t_m], 1)

            for i in range(4):
                sum_t1[i, t_m] = weights(Sigma[i, t_m:t_n+1, t_m] * G_0[i, t_n, t_m:t_n+1])  # sum[:, t2=t_m]

    for t_n in range(len(t)):
        for t_m in range(t_n+1, len(t), +1):
                G[:, t_n, t_m] = np.conj(G[:, t_m, t_n])

    print('-'*100)
    print('Finished calculation of bold propagators after', datetime.now() - start)
    print('\n')

    ########## Computation of Vertex Functions including hybridization lines between the upper and lower branch ##########

    # for every initial state i there is a set of 4 coupled equations for K
    for i in [1]:

        # indices are initial state | contour times on lower and upper branch
        K = np.zeros((4, 4, len(t), len(t)), complex)

        K_0 = np.zeros((4, 4, len(t), len(t)), complex)

        # indices are final state | lower and upper branch time
        Sigma = np.zeros((4, len(t), len(t)), complex)

        for f in range(4):
            K_0[i, f] = delta(i, f) * np.conj(G[i, :, None, 0]) * G[i, None, :, 0]

        for t_n in range(len(t)):
            sum_t1 = np.zeros((4, len(t)), complex)
            for t_m in range(len(t)):

                sum_t2 = np.zeros(4, complex)
                M = np.eye(4, dtype=complex)

                for f in range(4):
                    sum_t2[f] = dt**2 * np.trapz(G[f, t_m, :t_m+1] * sum_t1[f, :t_m+1])
                    # sum_t2[f] = dt ** 2 * weights(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

                    M[f] -= dt**2*np.sum(DeltaMatrix[f, :, t_n, t_m]*np.conj(G[:, t_n, t_n])*G[:, t_m, t_m], 0) * 1/4
                    # M[f] -= dt**2 * np.sum(DeltaMatrix[f, :, t_n, t_m] * np.conj(G[:, t_n, t_n]) * G[:, t_m, t_m], 0) * w(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

                # Dyson equation for time (t_m, t_n)
                K[i, :, t_n, t_m] = np.linalg.solve(M, K_0[i, :, t_n, t_m] + sum_t2)

                # Compute self-energy for time (t_m, t_n)
                SelfEnergy(K[i, :, t_n, t_m], DeltaMatrix[:, :, t_n, t_m], Sigma[:, t_n, t_m])

                for f in range(4):
                    sum_t1[f, t_m] = np.trapz(Sigma[f, :t_n+1, t_m] * np.conj(G[f, t_n, :t_n+1]))  # t_m = t_2
                    # sum_t1[f, t_m] = weights(Sigma[f, :t_n+1, t_m] * np.conj(G[f, t_n, :t_n+1]))  # t_m = t_2

        ########## Computation of two-times Green's functions ##########

        # Greens function with indices gtr/les | spin up/spin down | initial state | lower and upper branch time
        Green = np.zeros((2, 2, 4, len(t), len(t)), complex)

        for t1 in range(len(t)):
            for t_1 in range(len(t)):
                Green[0, 0, i, t1, t_1] = (K[i, 0, t1, t_1] * G[1, t1, t_1] + K[i, 2, t1, t_1] * G[3, t1, t_1])
                Green[1, 0, i, t1, t_1] = (K[i, 1, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[2, t1, t_1])
                Green[0, 1, i, t1, t_1] = (K[i, 0, t1, t_1] * G[2, t1, t_1] + K[i, 1, t1, t_1] * G[3, t1, t_1])
                Green[1, 1, i, t1, t_1] = (K[i, 2, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[1, t1, t_1])

        print('Finished calculation of K for initial state', i, 'after', datetime.now() - start)
        err = np.abs(1 - np.abs(np.sum(K[i, :, len(t)-1, len(t)-1], 0)))
        print('Error for inital state =', i, 'is', err)
        print('\n')

        print('1 - (Green_gtr + Green_les) for Spin Up site', i, 'is', 1 - np.real(Green[0, 0, i, len(t)-1, len(t)-1] + Green[1, 0, i, len(t)-1, len(t)-1]))
        print('1 - (Green_gtr + Green_les) for Spin Down site', i, 'is', 1 - np.real(Green[0, 1, i, len(t)-1, len(t)-1] + Green[1, 1, i, len(t)-1, len(t)-1]))

        print('\n')

        print('Population for Spin Up les on site', i, 'is', Green[1, 0, i, len(t) - 1, len(t) - 1])
        print('Population for Spin Down les on site', i, 'is', Green[1, 1, i, len(t) - 1, len(t) - 1])
        print('Population for Spin Up gtr on site', i, 'is', Green[0, 0, i, len(t) - 1, len(t) - 1])
        print('Population for Spin Down gtr on site', i, 'is', Green[0, 1, i, len(t) - 1, len(t) - 1])

        print('\n')

        with h5py.File(output, "a") as h5f:
            hdf5.save_green(h5f, gfsection, Green[:,:,1,:,:], (t,t))

def main():
    parser = argparse.ArgumentParser(description = "run nca impurity solver")
    parser.add_argument("--U",    type=float, default = 2.0)
    parser.add_argument("--hybfile",     default = "output.h5")
    parser.add_argument("--hybsection",  default = "hybridization/hyb")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
