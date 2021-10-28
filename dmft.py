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

def run_dmft(dim, U1, U2, T, pumpA, probeA, mu, v_0, tmax, dt, dw, tol, solver, phonon, fermion, Lambda, wC, t_diss_end, pumpOmega, t_pump_start, t_pump_end, probeOmega, t_probe_start, t_probe_end, lattice_structure, output, **kwargs):
    t  = np.arange(0, tmax, dt)

    msg = 'Starting DMFT loop for dim = {} | U1 = {} | U2={} | phonon = {} | fermion = {} | time = {} | dt = {}'.format(dim, U1, U2, T, phonon, fermion, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()

    # gf indices: [site, gtr/les, up/down, time, time]
    Green     = np.zeros((4, 2, 2, len(t), len(t)), complex)
    Green_old = np.zeros((4, 2, 2, len(t), len(t)), complex)
    Delta = np.zeros((4, 2, 2, len(t), len(t)), complex)

    U = np.zeros((4), float)
    U[:2] = U1
    U[2:] = U2
    # calculate and load bare propagators
    bareprop.bare_prop(t, U, Uconst=U)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    # delta indices: [gtr/les, up/down, time, time]
    hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    # timedep_hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    loaded = np.load('Delta.npz')
    for site in range(4):
        Delta[site] = loaded['D']

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

    # coupling to the dissipation bath can be turned off
    Lambda = coupling_to_diss_bath.gen_timedepLambda(t, t_diss_end, Lambda)

    # option of turning on a pump field and/or a probe field
    # v = electricField.genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t, lattice_structure)
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

        Green_old[:] = Green[:]
        # print('Green_old= ',Green_old)
        # print('Green= ',Green)

        for site in range(4):
            Green[site] = Solver(U[site], T, G_0[site], tmax, dt, Delta[site], phonon, fermion, Lambda, dissBath, iteration, output)

        # print(Green_old==Green)
        diff = np.amax(np.abs(Green_old - Green))
        # diff = 0

        interf_scaling = 1/(dim*2)
        bulk_scaling = 1-interf_scaling
        # antiferromagnetic self-consistency
        # site 0:
        Delta[0, :, 0] = v*bulk_scaling * Green[0, :, 1] + v*interf_scaling * Green[1, :, 1]
        Delta[0, :, 1] = v*bulk_scaling * Green[0, :, 0] + v*interf_scaling * Green[1, :, 0]

        # site 1:
        Delta[1, :, 0] = v*bulk_scaling * Green[0, :, 1] + v*interf_scaling * Green[2, :, 1]
        Delta[1, :, 1] = v*bulk_scaling * Green[0, :, 0] + v*interf_scaling * Green[2, :, 0]

        # site 2:
        Delta[2, :, 0] = v*interf_scaling * Green[1, :, 1] + v*bulk_scaling * Green[3, :, 1]
        Delta[2, :, 1] = v*interf_scaling * Green[1, :, 0] + v*bulk_scaling * Green[3, :, 0]

        # site 3:
        Delta[3, :, 0] = v*interf_scaling * Green[2, :, 1] + v*bulk_scaling * Green[3, :, 1]
        Delta[3, :, 1] = v*interf_scaling * Green[2, :, 0] + v*bulk_scaling * Green[3, :, 0]

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
    parser.add_argument("--params",   default = "runs.toml")
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
