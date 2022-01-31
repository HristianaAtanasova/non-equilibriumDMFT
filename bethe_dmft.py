# !/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
# import h5py
import toml

from datetime import datetime
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

import nca
import bareprop
import hybridization
import timedep_hybridization
import phonon_bath
import fermion_bath
import coupling_to_diss_bath
import electric_field
import constant_electric_field
import arrange_times
# import hdf5


def runNCA(U, pumpA, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output):
    t  = np.arange(0, tmax, dt)

    # hybsection = "dmft/iterations/{}/delta".format(iteration)
    # gfsection  = "dmft/iterations/{}/gf".format(iteration)

    # with h5py.File(output, "a") as h5f:
    #     hdf5.save_green(h5f, hybsection, Delta, (t,t))

    Green = nca.solve(t, U, pumpA, T, G_0, phonon, fermion, Lambda, dissBath, output, Delta)

    # with h5py.File(output, "r") as h5f:
    #     Green, _ = hdf5.load_green(h5f, gfsection)

    return Green

def runInch(U, tmax, dt, Delta):
    pass

def run_dmft(U, T, pumpA, probeA, mu, v_0, tmax, dt, dw, tol, solver, phonon, fermion, Lambda, wC, t_diss_end, pumpOmega, t_pump_start, t_pump_end, probeOmega, t_probe_start, t_probe_end, lattice_structure, output, **kwargs):
    t  = np.arange(0, tmax, dt)

    msg = 'Starting DMFT loop for U = {} | Temperature = {} | mu = {} | phonon = {} | fermion = {} | Lambda = {} | time = {} | dt = {}'.format(U, T, mu, phonon, fermion, Lambda, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()

    # gf indices: [gtr/les, up/down, time, time]
    Green = np.zeros((2, 2, len(t), len(t)), complex)
    Delta = np.zeros((2, 2, len(t), len(t)), complex)
    Green_old = np.zeros((2, 2, len(t), len(t)), complex)

    Uconst = U
    # calculate and load bare propagators
    bareprop.bare_prop(t, U, Uconst)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    # delta indices: [gtr/les, up/down, time, time]
    hybridization.genSemicircularHyb(T, mu, v_0, wC, tmax, dt, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    # timedep_hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    loaded = np.load('Delta.npz')
    Delta = loaded['D']

    if phonon == 1:
        phonon_bath.genPhononBath(t, mu, T)
        loaded = np.load('PhononBath.npz')
        dissBath = loaded['P']
    elif fermion == 1:
        fermion_bath.genFermionBath(T, mu, tmax, dt, dw, wC)
        loaded = np.load('FermionBath.npz')
        dissBath = loaded['F']
    else:
        dissBath = 0

    # coupling to the dissipation bath can be turned off
    Lambda = coupling_to_diss_bath.gen_timedepLambda(t, t_diss_end, Lambda)

    # option of turning on a pump field and/or a probe field
    # v = electric_field.genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t, lattice_structure)
    v = constant_electric_field.genv(pumpA, v_0, t, lattice_structure)
    # v = v_0

    # set solver
    if solver == 0:
        Solver = runNCA
    elif solver == 1:
        Solver = runInch
    else:
        raise Exception("solver {:s} not recognized".format(solver))

    msg = 'Start DMFT iteration until the impurity Greens function has converges to the lattice Greens function:'
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    # DMFT self-consistency loop
    diff = np.inf
    iteration = 0
    while diff > tol:
        iteration += 1

        Green_old = Green

        # plt.plot(t, np.real(Delta[0, 0, ::-1, len(t)-1]), '-', t, np.imag(Delta[0, 0, ::-1, len(t)-1]), '--', label = 'Hyp_gtr')
        # plt.plot(t, np.real(Delta[1, 0, ::-1, len(t)-1]), '-', t, np.imag(Delta[1, 0, ::-1, len(t)-1]), '--', label = 'Hyb_les')
        # plt.legend()
        # plt.savefig('hybridization_bethe.pdf')
        # plt.close()

        Green = Solver(U, pumpA, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output)

        diff = np.amax(np.abs(Green_old - Green))
        # diff = 0

        # antiferromagnetic self-consistency
        Delta[:, 1] = v * Green[:, 0]
        Delta[:, 0] = v * Green[:, 1]

        # # doubly occupied and empty sublattice
        # Delta[0, :] = v * Green[1, :]
        # Delta[1, :] = v * Green[0, :]

        contour_Delta = arrange_times.real_time_to_keldysh(Delta, t, tmax)
        contour_Green = arrange_times.real_time_to_keldysh(Green, t, tmax)

        Delta = arrange_times.keldysh_to_real_time(contour_Delta, t, tmax)
        Green = arrange_times.keldysh_to_real_time(contour_Green, t, tmax)

        plt.matshow(contour_Delta[0].real)
        plt.colorbar()
        plt.savefig('bethe_Delta_real.pdf')
        plt.close()

        plt.matshow(contour_Delta[0].imag)
        plt.colorbar()
        plt.savefig('bethe_Dalta_imag.pdf')
        plt.close()

        plt.matshow(contour_Green[0].real)
        plt.colorbar()
        plt.savefig('bethe_Green_real.pdf')
        plt.close()

        plt.matshow(contour_Green[0].imag)
        plt.colorbar()
        plt.savefig('bethe_Green_imag.pdf')
        plt.close()

        print('\n')
        msg = 'U = {}, iteration {}: diff = {} (elapsed time = {})'
        print(msg.format(U, iteration, diff, datetime.now() - start))
        print('-'*len(msg)*2)

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
