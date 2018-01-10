# !/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import toml

from datetime import datetime
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

import nca
import bareprop
import hybridization
import timedep_hybridization
import phononBath
import fermionBath
import electricField
import hdf5


def runNCA(U, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output):
    t  = np.arange(0, tmax, dt)

    hybsection = "dmft/iterations/{}/delta".format(iteration)
    gfsection  = "dmft/iterations/{}/gf".format(iteration)

    with h5py.File(output, "a") as h5f:
        hdf5.save_green(h5f, hybsection, Delta, (t,t))

    nca.solve(t, U, G_0, phonon, fermion, Lambda, dissBath, output, hybsection, gfsection)

    with h5py.File(output, "r") as h5f:
        Green, _ = hdf5.load_green(h5f, gfsection)

    return Green

def runInch(U, tmax, dt, Delta):
    pass

def run_dmft(U, Uconst, T, pumpA, probeA, mu, v_0, tmax, dt, dw, tol, solver, phonon, fermion, Lambda, pumpOmega, probeOmega, output, **kwargs):
    t  = np.arange(0, tmax, dt)

    msg = 'Starting DMFT loop for U = {} | Uconst = {} | Temperature = {} | mu = {} | phonon = {} | fermion = {} | Lambda = {} | time = {} | dt = {}'.format(U, Uconst, T, mu, phonon, fermion, Lambda, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()

    # gf indices: [gtr/les, up/down, state, time, time]
    Green     = np.zeros((2, 2, len(t), len(t)), complex)
    Green_old = np.zeros((2, 2, len(t), len(t)), complex)

    # calculate and load bare propagators
    bareprop.bare_prop(t, U, Uconst)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    # delta indices: [gtr/les, up/down, time, time]
    hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    # timedep_hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    loaded = np.load('Delta.npz')
    Delta = loaded['D']

    if phonon == 1:
        phononBath.genPhononBath(t, mu, T)
        loaded = np.load('PhononBath.npz')
        dissBath = loaded['P']
    elif fermion == 1:
        fermionBath.genFermionBath(T, mu, tmax, dt, dw)
        loaded = np.load('FermionBath.npz')
        dissBath = loaded['F']
    else:
        dissBath = 0

    # option of turning on a pump field and/or a probe field
    v = electricField.genv(pumpA, pumpOmega, probeA, probeOmega, v_0, t)

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

        Green_old = Green

        Green = Solver(U, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output)

        diff = np.amax(np.abs(Green_old - Green))
        # diff = 0

        # antiferromagnetic self-consistency
        Delta[:, 0] = v * Green[:, 1]
        Delta[:, 1] = v * Green[:, 0]

        msg = 'U = {}, iteration {}: diff = {} (elapsed time = {})'
        print(msg.format(U, iteration, diff, datetime.now() - start))

    msg = 'Computation finished after {} iterations and {} seconds'.format(iteration, datetime.now() - start)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    return Green

def main():
    parser = argparse.ArgumentParser(description = "run dmft")
    parser.add_argument("--output",   default = "output.h5")
    # parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))

    with h5py.File(args.output, "a") as h5f:
        for k,v in params.items():
            h5f.create_dataset("dmft/params/" + k, data=v)

    Green = run_dmft(**params)

if __name__ == "__main__":
    main()
