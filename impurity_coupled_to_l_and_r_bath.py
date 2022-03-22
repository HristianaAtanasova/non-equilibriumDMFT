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


def runNCA(U, mu1, mu2, pumpA, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output):
    t  = np.arange(0, tmax, dt)

    # hybsection = "dmft/iterations/{}/delta".format(iteration)
    # gfsection  = "dmft/iterations/{}/gf".format(iteration)

    # with h5py.File(output, "a") as h5f:
    #     hdf5.save_green(h5f, hybsection, Delta, (t,t))

    Green = nca.solve(t, U, mu1, mu2, pumpA, T, G_0, phonon, fermion, Lambda, dissBath, output, Delta)

    # with h5py.File(output, "r") as h5f:
    #     Green, _ = hdf5.load_green(h5f, gfsection)

    return Green

def runInch(U, tmax, dt, Delta):
    pass

def run_dmft(U, T, pumpA, probeA, mu1, mu2, v_0, tmax, dt, dw, tol, solver, phonon, fermion, Lambda, wC, t_diss_end, pumpOmega, t_pump_start, t_pump_end, probeOmega, t_probe_start, t_probe_end, lattice_structure, output, **kwargs):
    t  = np.arange(0, tmax, dt)

    msg = 'Starting DMFT loop for U = {} | Temperature = {} | mu1 = {} | mu2 = {} | phonon = {} | fermion = {} | Lambda = {} | time = {} | dt = {}'.format(U, T, mu1, mu2, phonon, fermion, Lambda, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()

    # gf indices: [gtr/les, up/down, time, time]
    Green = np.zeros((2, 2, len(t), len(t)), complex)
    Delta = np.zeros((2, 2, len(t), len(t)), complex)

    Uconst = U
    # calculate and load bare propagators
    bareprop.bare_prop(t, U, Uconst)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    # delta indices: [gtr/les, up/down, time, time]
    # hybridization.genSemicircularHyb(T, mu1, v_0, wC, tmax, dt, dw)
    hybridization.genWideBandHyb(T, mu1, wC, tmax, dt, dw)
    loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(mu1, T, dt))
    Delta_left = loaded['D']

    # hybridization.genSemicircularHyb(T, mu2, v_0, wC, tmax, dt, dw)
    hybridization.genWideBandHyb(T, mu2, wC, tmax, dt, dw)
    loaded = np.load('Delta_mu={}_T={}_dt={}.npz'.format(mu2, T, dt))
    Delta_right = loaded['D']

    if phonon == 1:
        phonon_bath.genPhononBath(t, mu1, T)
        loaded = np.load('PhononBath.npz')
        dissBath = loaded['P']
    elif fermion == 1:
        fermion_bath.genFermionBath(T, mu1, tmax, dt, dw, wC)
        loaded = np.load('FermionBath.npz')
        dissBath = loaded['F']
    else:
        dissBath = 0

    # coupling to the dissipation bath can be turned off
    Lambda = coupling_to_diss_bath.gen_timedepLambda(t, t_diss_end, Lambda)

    # option of turning on a pump field and/or a probe field
    # v = electric_field.genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t, lattice_structure)
    # v = constant_electric_field.genv(pumpA, v_0, t, lattice_structure)
    v = v_0

    # set solver
    if solver == 0:
        Solver = runNCA
    elif solver == 1:
        Solver = runInch
    else:
        raise Exception("solver {:s} not recognized".format(solver))

    iteration = 0 

    Delta = (Delta_left + Delta_right) / 2.0

    Green = Solver(U, mu1, mu2, pumpA, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output)

    return Green

def main():
    parser = argparse.ArgumentParser(description = "run dmft")
    # parser.add_argument("--output",   default = "output.h5")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run_impurity.toml")
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
