#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import toml

from datetime import datetime
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

import nca
import hybridization
import hdf5

def runNCA(U, tmax, dt, Delta, iteration, output):
    t  = np.arange(0, tmax, dt)
    U_ = U * np.ones(t.shape[0], float)

    hybsection = "dmft/iterations/{}/delta".format(iteration)
    gfsection  = "dmft/iterations/{}/gf".format(iteration)

    with h5py.File(output, "a") as h5f:
        hdf5.save_green(h5f, hybsection, Delta, (t,t))

    nca.solve(t, U_, output, hybsection, gfsection)

    with h5py.File(output, "r") as h5f:
        Green, _ = hdf5.load_green(h5f, gfsection)

    return Green

def runInch(U, tmax, dt, Delta):
    pass

def run_dmft(U, T, mu, v_0, tmax, dt, dw, tol, solver, output, **kwargs):
    t  = np.arange(0, tmax, dt)

    msg = 'Starting DMFT loop for U = {} | Temperature = {} | time = {} | dt = {}'.format(U, T, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()

    # gf indices: [gtr/les, up/down, state, time, time]
    Green     = np.zeros((2, 2, len(t), len(t)), complex)
    Green_old = np.zeros((2, 2, len(t), len(t)), complex)

    # delta indices: [gtr/les, up/down, time, time]
    np.seterr(over="ignore")
    Delta = hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, dw)
    np.seterr(over="warn")

    if solver == "NCA":
        Solver = runNCA
    elif solver == "inchworm":
        Solver = runInch
    else:
        raise Exception("solver {:s} not recognized".format(solver))

    # DMFT self-consistency loop
    diff = np.inf
    iteration = 0
    while diff > tol:
        iteration += 1

        Green_old = Green

        Green = Solver(U, tmax, dt, Delta, iteration, output)

        diff = np.amax(np.abs(Green_old - Green))

        # antiferromagnetic self-consistency
        Delta[:, 0] = v_0 * Green[:, 1] * v_0
        Delta[:, 1] = v_0 * Green[:, 0] * v_0

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
    parser.add_argument("--params",   default = "params.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))

    with h5py.File(args.output, "a") as h5f:
        for k,v in params.items():
            h5f.create_dataset("dmft/params/" + k, data=v)

    Green = run_dmft(**params)

    if args.savetxt:
        gtr_up   = 'gtr_up_U={}_T={}_t={}_dt={}.out'
        les_up   = 'les_up_U={}_T={}_t={}_dt={}.out'
        gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}.out'
        les_down = 'les_down_U={}_T={}_t={}_dt={}.out'

        np.savetxt(gtr_up.format(*[params[x] for x in ["U", "T", "tmax", "dt"]]), Green[0, 0, 1].view(float), delimiter=' ')
        np.savetxt(les_up.format(*[params[x] for x in ["U", "T", "tmax", "dt"]]), Green[1, 0, 1].view(float), delimiter=' ')
        np.savetxt(gtr_down.format(*[params[x] for x in ["U", "T", "tmax", "dt"]]), Green[0, 1, 1].view(float), delimiter=' ')
        np.savetxt(les_down.format(*[params[x] for x in ["U", "T", "tmax", "dt"]]), Green[1, 1, 1].view(float), delimiter=' ')

if __name__ == "__main__":
    main()
