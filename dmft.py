# !/usr/bin/env python3
import numpy as np
import scipy
# from numpy.linalg import inv, pinv
from scipy.linalg import inv, pinv, pinvh
import matplotlib.pyplot as plt
import argparse
# import h5py
import toml

from datetime import datetime
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

import nca
import bareprop
import k_greensfunction 
import hybridization
import timedep_hybridization
import phononBath
import fermionBath
import coupling_to_diss_bath
import electricField
# import hdf5


def runNCA(U, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output):
    t  = np.arange(0, tmax, dt)

    # hybsection = "dmft/iterations/{}/delta".format(iteration)
    # gfsection  = "dmft/iterations/{}/gf".format(iteration)

    # with h5py.File(output, "a") as h5f:
    #     hdf5.save_green(h5f, hybsection, Delta, (t,t))

    Green = nca.solve(t, U, T, G_0, phonon, fermion, Lambda, dissBath, output, Delta)

    # with h5py.File(output, "r") as h5f:
    #     Green, _ = hdf5.load_green(h5f, gfsection)

    return Green

def runInch(U, tmax, dt, Delta):
    pass

def run_dmft(U, T, pumpA, probeA, mu, v_0, tmax, dt, dw, tol, solver, phonon, fermion, Lambda, wC, t_diss_end, pumpOmega, t_pump_start, t_pump_end, probeOmega, t_probe_start, t_probe_end, lattice_structure, output, **kwargs):
    t = np.arange(0, tmax, dt)
    w = np.arange(-wC, wC, dw) 

    msg = 'Starting DMFT loop for U = {} | Temperature = {} | mu = {} | phonon = {} | fermion = {} | Lambda = {} | time = {} | dt = {}'.format(U, T, mu, phonon, fermion, Lambda, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()

    # gf indices: [gtr/les, up/down, time, time]
    Green = np.zeros((2, 2, len(t), len(t)), complex)
    Green_old = np.zeros((2, 2, len(t), len(t)), complex)
    Delta = np.zeros((2, 2, len(t), len(t)), complex)
    Delta_old = np.zeros((2, 2, len(t), len(t)), complex)
    Weiss = np.zeros((2, 2, len(t), len(t)), complex)
    Sigma = np.zeros((2, 2, len(t), len(t)), complex)

    Uconst = U
    # calculate and load bare propagators
    bareprop.bare_prop(t, U, Uconst)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    k_greensfunction.gen_k_dep_Green(T, v_0, U, mu, tmax, dt, wC, dw)
    loaded = np.load('Green_k_inv.npz')
    Green_k_inv = loaded['G']
    Green_d = loaded['G_d']

    # delta indices: [gtr/les, up/down, time, time]
    hybridization.genGaussianHyb(T, mu, v_0, tmax, dt, wC, dw)
    # hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    # timedep_hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    loaded = np.load('Delta.npz')
    Delta = loaded['D']
    dos = loaded['dos']

    if phonon == 1:
        phononBath.genPhononBath(t, mu, T)
        loaded = np.load('PhononBath.npz')
        dissBath = loaded['P']
    elif fermion == 1:
        fermionBath.genFermionBath(T, mu, tmax, dt, dw, wC)
        loaded = np.load('FermionBath.npz')
        dissBath = loaded['F']
    else:
        dissBath = 0

    ## coupling to the dissipation bath can be turned off
    # Lambda = coupling_to_diss_bath.gen_timedepLambda(t, t_diss_end, Lambda)

    # option of turning on a pump field and/or a probe field
    #v = electricField.genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t, lattice_structure)
    v = v_0

    # set solver
    if solver == 0:
        Solver = runNCA
    elif solver == 1:
        Solver = runInch
    else:
        raise Exception("solver {:s} not recognized".format(solver))

    # DMFT self-consistency loop
    diff = np.inf
    iteration = 0
    while diff > tol:
        iteration += 1

        Green_old[:] = Green
        Delta_old[:] = Delta
        Green = Solver(U, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output)

        plt.plot(t, np.real(Delta[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Delta[0, 0, ::-1, len(t)-1]), '--', label = 'Hyp_gtr')
        plt.plot(t, np.real(Delta[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Delta[1, 0, ::-1, len(t)-1]), '--', label = 'Hyb_les')
        plt.legend()
        plt.savefig('hybridization.pdf')
        plt.close()

        plt.plot(t, np.real(Green[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Green[0, 0, ::-1, len(t)-1]), '--', label = 'Green_gtr')
        plt.plot(t, np.real(Green[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Green[1, 0, ::-1, len(t)-1]), '--', label = 'Green_les')
        plt.legend()
        plt.savefig('green.pdf')
        plt.close()

        # print('Delta _gtr = ', (Delta[0, 0]))
        # print('det_Delta_gtr = ', np.linalg.det(Delta[0, 0]))

        # print('Delta _les = ', (Delta[1, 0]))
        # print('det_Delta_les = ', np.linalg.det(Delta[1, 0]))

        print('Green_gtr = ', (Green[0, 0]))
        print('det_Green_gtr = ', np.linalg.det(Green[0, 0]))
        print('inv_Green_gtr = ', inv(Green[0, 0]))
        # print('inv_Green_gtr * Green_gtr = ', np.matmul(inv(Green[0, 0]), Green[0, 0]))
        # print('pinv_Green_gtr * Green_gtr = ', np.matmul(pinv(Green[0, 0]), Green[0, 0]))

        print('Green_les = ', Green[1, 0])
        print('det_Green_les = ', np.linalg.det(Green[1, 0]))
        print('inv_Green_les = ', inv(Green[1, 0]))

        for comp in range(2):
           for spin in range(2):
               print('comp = ', comp, 'spin = ', spin)
               Sigma[comp, spin] = inv(Delta[comp, spin]) - inv(Green[comp, spin])
        
        # plt.matshow(np.real(Green[0,0]))
        # plt.savefig('G_matrix_real.pdf')
        # plt.close()

        # plt.matshow(np.imag(Green[0,0]))
        # plt.savefig('G_matrix_imag.pdf')
        # plt.close()

        M = Green_k_inv - Sigma
        Green_local = np.zeros((2, 2, len(t), len(t)), complex)
        for comp in range(2):
            for spin in range(2):
                Green_local[comp, spin] = inv(M[comp, spin])

        I = np.zeros((2, 2, len(t), len(t)), complex)
        for comp in range(2):
            for spin in range(2):
                I[comp, spin] = inv(Green_local[comp, spin]) 

        I = I + Sigma
        for comp in range(2):
            for spin in range(2):
                Weiss[comp, spin] = (inv(I[comp, spin])) 
                Delta[comp, spin] = - Weiss[comp, spin] + Green_d[comp, spin] 
                # Delta[comp, spin] = + Weiss[comp, spin] + Green_d[comp, spin] 

        # Delta = Delta_old / 2.0 + Delta / 2.0
        Delta[:, 0] = Delta[:, 1]
        Delta[:, 1] = Delta[:, 0]

        plt.plot(t, np.real(Weiss[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Weiss[0, 0, ::-1, len(t)-1]), '--', label = 'Weiss_gtr')
        plt.plot(t, np.real(Weiss[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Weiss[1, 0, ::-1, len(t)-1]), '--', label = 'Weiss_les')
        plt.legend()
        plt.savefig('weiss_field.pdf')
        plt.close()

        plt.plot(t, np.real(Green_local[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Green_local[0, 0, ::-1, len(t)-1]), '--', label = 'Green_local_gtr')
        plt.plot(t, np.real(Green_local[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Green_local[1, 0, ::-1, len(t)-1]), '--', label = 'Green_local_les')
        plt.legend()
        plt.savefig('green_local.pdf')
        plt.close()

        plt.plot(t, np.real(Green_d[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Green_d[0, 0, ::-1, len(t)-1]), '--', label = 'Green_d_gtr')
        plt.plot(t, np.real(Green_d[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Green_d[1, 0, ::-1, len(t)-1]), '--', label = 'Green_d_les')
        plt.legend()
        plt.savefig('green_dot.pdf')
        plt.close()

        # plt.plot(t, np.real(Delta[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Delta[0, 0, ::-1, len(t)-1]), '--', label = 'Delta_gtr')
        # plt.plot(t, np.real(Delta[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Delta[1, 0, ::-1, len(t)-1]), '--', label = 'Delta_les')
        # plt.legend()
        # plt.savefig('delta.pdf')
        # plt.close()

        # plt.plot(t, np.real(Sigma[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Sigma[0, 0, ::-1, len(t)-1]), '--', label = 'Sigma_gtr')
        # plt.plot(t, np.real(Sigma[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Sigma[1, 0, ::-1, len(t)-1]), '--', label = 'Sigma_les')
        # plt.legend()
        # plt.savefig('sigma.pdf')
        # plt.close()

        diff = np.amax(np.abs(Green_old - Green))
        diff = 0

        msg = 'U = {}, iteration {}: diff = {} (elapsed time = {})'
        print(msg.format(U, iteration, diff, datetime.now() - start))

    msg = 'Computation finished after {} iterations and {} seconds'.format(iteration, datetime.now() - start)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    return Green

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

    # with h5py.File(args.output, "a") as h5f:
    #     for k,v in params.items():
    #         h5f.create_dataset("dmft/params/" + k, data=v)

    Green = run_dmft(**params)

if __name__ == "__main__":
    main()
