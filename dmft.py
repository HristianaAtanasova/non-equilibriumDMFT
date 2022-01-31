# !/usr/bin/env python3
import numpy as np
import scipy
# from numpy.linalg import inv
from scipy.linalg import inv
import matplotlib.pyplot as plt
import argparse
import toml

from datetime import datetime
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

import nca
import bareprop
import k_greensfunction 
import hybridization
import timedep_hybridization
import phonon_bath
import fermion_bath
import coupling_to_diss_bath
import electric_field
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
    t = np.arange(0, tmax, dt)
    t_contour = np.arange(-tmax, tmax, dt)
    w = np.arange(-wC, wC, dw) 

    msg = 'Starting DMFT loop for U = {} | Temperature = {} | mu = {} | phonon = {} | fermion = {} | Lambda = {} | time = {} | dt = {}'.format(U, T, mu, phonon, fermion, Lambda, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    # gf indices: [gtr/les, up/down, time, time]
    Green = np.zeros((2, 2, len(t), len(t)), complex)
    Green_old = np.zeros((2, 2, len(t), len(t)), complex)
    Delta = np.zeros((2, 2, len(t), len(t)), complex)
    Green_local = np.zeros((2, 2, len(t), len(t), len(w)), complex)
    Green_lattice = np.zeros((2, len(t_contour), len(t_contour), len(w)), complex)
    contour_Weiss = np.zeros((2, len(t_contour), len(t_contour)), complex)
    contour_Sigma = np.zeros((2, len(t_contour), len(t_contour)), complex)

    Uconst = U
    # calculate and load bare propagators
    bareprop.bare_prop(t, U, Uconst)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    k_greensfunction.gen_k_dep_Green(T, v_0, U, mu, tmax, dt, wC, dw)
    loaded = np.load('Green_k_inv.npz')
    contour_Green_k_inv = loaded['G_k_inv']
    contour_Green_k = loaded['G_k']
    contour_Green_dot = loaded['G_d']

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

    ## coupling to the dissipation bath can be turned off
    # Lambda = coupling_to_diss_bath.gen_timedepLambda(t, t_diss_end, Lambda)

    # option of turning on a pump field and/or a probe field
    #v = electric_field.genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t, lattice_structure)
    v = v_0

    # set solver
    if solver == 0:
        Solver = runNCA
    elif solver == 1:
        Solver = runInch
    else:
        raise Exception("solver {:s} not recognized".format(solver))

    # delta indices: [gtr/les, up/down, time, time]
    # hybridization.genGaussianHyb(T, mu, v_0, tmax, dt, wC, dw)
    hybridization.genSemicircularHyb(T, mu, v_0, wC, tmax, dt, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    # timedep_hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    loaded = np.load('Delta.npz')
    Delta = loaded['D']
    dos = loaded['dos']

    msg = 'Start DMFT iteration until the impurity Greens function has converges to the lattice Greens function:'
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()
    diff = np.inf
    iteration = 0
    tol = 0.01
    plotting = 1 
    while diff > tol:
        iteration += 1

        Delta_old = Delta
        Green_old = Green

        Green = Solver(U, pumpA, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output)
        
        contour_Delta = arrange_times.real_time_to_keldysh(Delta, t, tmax)
        contour_Green = arrange_times.real_time_to_keldysh(Green, t, tmax)
        
        if iteration == 1:
            Delta_init = contour_Delta

        if plotting == 1:
            plt.matshow(contour_Delta[0].real)
            plt.colorbar()
            plt.savefig('{}_Delta_real_up.pdf'.format(iteration))
            plt.close()

            plt.matshow(contour_Delta[0].imag)
            plt.colorbar()
            plt.savefig('{}_Dalta_imag_up.pdf'.format(iteration))
            plt.close()

            plt.matshow(contour_Delta[1].real)
            plt.colorbar()
            plt.savefig('{}_Delta_real_down.pdf'.format(iteration))
            plt.close()

            plt.matshow(contour_Delta[1].imag)
            plt.colorbar()
            plt.savefig('{}_Dalta_imag_down.pdf'.format(iteration))
            plt.close()

        for spin in range(2):
            # I = (contour_Delta[spin] - contour_Green_dot[spin])
            # contour_Sigma[spin] = - inv(I) + inv(contour_Green[spin])

            contour_Sigma[spin] = inv(-contour_Delta[spin] + contour_Green[spin])

        contour_Green_local = np.zeros((2, len(t_contour), len(t_contour), len(w)), complex)
        for w1 in range(len(w)):        
            for spin in range(2):
                Green_lattice[spin, :, :, w1] = inv(contour_Green_k_inv[spin, :, :, w1]) * dos[w1]
  
                # M = contour_Green_k_inv[spin, :, :, w1] + (contour_Sigma[spin])
                # contour_Green_local[spin, :, :, w1] = inv(M) * dos[w1]

                contour_Green_local[spin, :, :, w1] = (contour_Green_k[spin, :, :, w1] + inv(contour_Sigma[spin])) * dos[w1]

        # G_local = dw * np.trapz(contour_Green_local, x = w, axis = -1)
        G_local = dw * np.sum(contour_Green_local, axis = -1)

        G_local_inv = inv(G_local[0]) 
        G_local_0 = dw * np.sum(Green_lattice, axis = -1)
        G_local_0_inv = inv(G_local_0[0]) 

        for spin in range(2):
            I = inv(G_local[spin]) * (np.pi / 2.0) + contour_Sigma[spin]
            contour_Delta[spin] = inv(I) 

        Green = arrange_times.keldysh_to_real_time(contour_Green, t, tmax)
        Delta = arrange_times.keldysh_to_real_time(contour_Delta, t, tmax)

        # plt.plot(contour_Green[0, :len(t), (len(t))].real, label='Green_imp')
        # plt.plot(G_local_0[0, :len(t), (len(t))].real, label='Green_local_0')
        # plt.legend()
        # plt.savefig('cut_Green.pdf')
        # plt.close()

        Delta_ = np.copy(Delta)
        Delta[:, 0] = Delta_[:, 1]
        Delta[:, 1] = Delta_[:, 0]

        diff = np.amax(np.abs(Green_old - Green))
        # diff = 0

        inv_contour_Delta = inv(contour_Delta[0])
        inv_contour_Sigma = inv(contour_Sigma[0])
        inv_contour_Green = inv(contour_Green[0])
        inv_contour_Green_dot = inv(contour_Green_dot[0])

        if plotting == 1:
            # plt.matshow(inv_contour_Green_dot.real)
            # plt.colorbar()
            # plt.savefig('inv_Green_dot_real.pdf')
            # plt.close()

            # plt.matshow(inv_contour_Green_dot.imag)
            # plt.colorbar()
            # plt.savefig('inv_Green_dot_imag.pdf')
            # plt.close()

            # plt.matshow(G_local_0_inv.real)
            # plt.colorbar()
            # plt.savefig('inv_Green_local_0_real.pdf')
            # plt.close()

            # plt.matshow(G_local_0_inv.imag)
            # plt.colorbar()
            # plt.savefig('inv_Green_local_0_imag.pdf')
            # plt.close()

            plt.matshow(G_local_inv.real)
            plt.colorbar()
            plt.savefig('{}_inv_Green_local_real.pdf'.format(iteration))
            plt.close()

            plt.matshow(G_local_inv.imag)
            plt.colorbar()
            plt.savefig('{}_inv_Green_local_imag.pdf'.format(iteration))
            plt.close()

            plt.matshow(inv_contour_Sigma.real)
            plt.colorbar()
            plt.savefig('{}_inv_Sigma_real.pdf'.format(iteration))
            plt.close()

            plt.matshow(inv_contour_Sigma.imag)
            plt.colorbar()
            plt.savefig('{}_inv_Sigma_imag.pdf'.format(iteration))
            plt.close()

            plt.matshow(contour_Green[0].real)
            plt.colorbar()
            plt.savefig('{}_Green_real.pdf'.format(iteration))
            plt.close()

            plt.matshow(contour_Green[0].imag)
            plt.colorbar()
            plt.savefig('{}_Green_imag.pdf'.format(iteration))
            plt.close()

            plt.matshow(contour_Green_dot[0].real)
            plt.colorbar()
            plt.savefig('Green_dot_real.pdf')
            plt.close()

            plt.matshow(contour_Green_dot[0].imag)
            plt.colorbar()
            plt.savefig('Green_dot_imag.pdf')
            plt.close()

            plt.matshow(G_local[0].real)
            plt.colorbar()
            plt.savefig('{}_Green_local_real.pdf'.format(iteration))
            plt.close()

            plt.matshow(G_local[0].imag)
            plt.colorbar()
            plt.savefig('{}_Green_local_imag.pdf'.format(iteration))
            plt.close()

            plt.matshow(G_local_0[0].real)
            plt.colorbar()
            plt.savefig('Green_local_0_real.pdf')
            plt.close()

            plt.matshow(G_local_0[0].imag)
            plt.colorbar()
            plt.savefig('Green_local_0_imag.pdf')
            plt.close()

            plt.matshow(contour_Sigma[0].real)
            plt.colorbar()
            plt.savefig('{}_Sigma_real.pdf'.format(iteration))
            plt.close()

            plt.matshow(contour_Sigma[0].imag)
            plt.colorbar()
            plt.savefig('{}_Sigma_imag.pdf'.format(iteration))
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
