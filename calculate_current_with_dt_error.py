import argparse
import toml
import zipfile
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import os
import electric_field

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def calculate_av_current(I, tmax, period, dt):
    n_period =int(period/dt)
    t_start = int((tmax-period)/dt)
    avI = np.sum(I[t_start:])/n_period
    return avI

def calculate_error(avI_1, I, tmax, period, dt):
    n_period =int(period/dt)
    t_start = int((tmax-2*period)/dt)
    t_end = int((tmax-period)/dt)
    avI_2 = np.sum(I[t_start:t_end])/n_period
    return np.abs(avI_1 - avI_2)

Path = os.getcwd() + '/tol={}/U{}_probeA{}_probeOmega{}_pumpA{}_pumpOmega{}/T{}'

# set up parameters
tmax = 10
U_ = [10.0]
# pumpOmega_ = [5.0, 10.0, 15.0, 20.0]
pumpOmega_ = [10.0]
probeOmega_ = [1.25]
# pumpA_ = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.831, 4.5, 5.0, 5.5, 6.0, 6.5, 7.015, 7.5, 8.0]
pumpA_ = [0.0, 3.831]
probeA_ = [0.0]
Temperature_ = [1.0]
dt_ = [0.1, 0.05, 0.02, 0.01, 0.05]
wC = 50

# load energy hybridization function
ferm_Delta_E_name = 'ferm_Delta_dt={}_wC={}_E_T={}_tmax{}.npz'
ferm_Delta_E = np.zeros((len(Temperature_),2,2,len(t), len(t)), complex)
ferm_Delta_name = 'ferm_Delta_dt={}_wC={}_T={}_tmax{}.npz'
ferm_Delta = np.zeros((len(Temperature_),2,2,len(t), len(t)), complex)

av_I = np.zeros(len(pumpA_))
dev_dt = np.zeros((len(dt_),len(dt_)), float)
# calculate current
for U in U_:
    for T_ in range(len(Temperature_)):
        T = Temperature_[T_]
        for probeOmega in probeOmega_:
            for probeA in probeA_:
                for pumpOmega in pumpOmega_:
                    for dt_i in range(len(dt_)):
                        dt = dt_[dt_i]
                        t  = np.arange(0, tmax, dt)

                        loaded = np.load(ferm_Delta_E_name.format(dt, wC, T, tmax))
                        ferm_Delta_E[T_] = loaded['Delta']
                        loaded = np.load(ferm_Delta_name.format(dt, wC, T, tmax))
                        ferm_Delta[T_] = loaded['Delta']

                        for pumpA_i in range(len(pumpA_)):
                            pumpA = pumpA_[pumpA_i]
                            msg = 'Executing wC = {} |  tol = {} |  U = {} | pumpOmega={} | pumpA = {} | T = {}'
                            print(msg.format(wC, tol, U, pumpOmega, pumpA, T))

                            path = Path.format(tol, U, probeA, probeOmega, pumpA, pumpOmega, T)
                            parser = argparse.ArgumentParser(description = "run dmft")
                            parser.add_argument("--params",   default = path + "/run.toml")
                            args = parser.parse_args()

                            with open(args.params, "r") as f:
                                params = toml.load(f)

                            params.update(vars(args))

                            # load Greens functions
                            Green_path = path + '/Green_T={}.npz'
                            loaded = np.load(Green_path.format(T))
                            t = loaded['t']
                            dt = params['dt']
                            Green = loaded['Green']
                            period = 2*np.pi/pumpOmega
                            t_period = np.arange(params['tmax']-period,params['tmax'],dt)

                            # calculate energy current I_E and lattice current I_n
                            I_E_bath = np.zeros((2,len(t)), complex)                # energy current; spin up/spin down | time
                            I_N_bath = np.zeros((2,len(t)), complex)                # lattice current; spin up/spin down | time
                            I_N = np.zeros((2,len(t)), complex)                     # lattice current; spin up/spin down | time

                            v = electric_field.genv(pumpA, pumpOmega, params['t_pump_start'], params['t_pump_end'], probeA, probeOmega, params['t_probe_start'], params['t_probe_end'], params['v_0'], t, params['lattice_structure'])
                            Delta = np.zeros((2, 2, len(t), len(t)), complex) # dynamical mean field
                            Delta[:, 0] = v * Green[:, 1]
                            Delta[:, 1] = v * Green[:, 0]

                            for t1 in range(len(t)):
                                # # calculate energy current between site and dissipatative bath
                                I_E_bath[0,t1] = dt*np.trapz(ferm_Delta_E[T_,0,0,:t1,t1]*G[0,0,:t1,t1])
                                I_E_bath[1,t1] = dt*np.trapz(ferm_Delta_E[T_,0,1,:t1,t1]*G[0,1,:t1,t1])
                                #
                                # # calculate particle current
                                # I_N_bath[0,t1] = dt*np.trapz(ferm_Delta[T_,0,0,:t1,t1]*G[0,0,:t1,t1])
                                # I_N_bath[1,t1] = dt*np.trapz(ferm_Delta[T_,0,1,:t1,t1]*G[0,1,:t1,t1])
                                #
                                # # calculate particle current in the lattice
                                # I_N[0,t1] = dt*np.trapz(Delta[0,0,t1,:t1]*G[0,0,t1,:t1])
                                # I_N[1,t1] = dt*np.trapz(Delta[0,1,t1,:t1]*G[0,1,t1,:t1])

                            # plt.plot(t, np.real(I_E_bath[0]+I_E_bath[1]), label='tau,t1_real')
                            # plt.plot(t, np.imag(I_E_bath[0]+I_E_bath[1]), label='tau,t1_imag')

                            av_I[pumpA_i] = calculate_av_current(np.real(I_E_bath[0,0:len(t)]+I_E_bath[1,0:len(t)]), params['tmax'], period, dt)
                            dev_dt[dt_i, pumpA_i] = av_I[pumpA_i]-0

                        # plt.plot(pumpA_, av_I, marker='o', label='wC={}'.format(wC))

                    plt.plot(dt_, dev_dt[:, 0], marker='o', label='pumpA=0.0')
                    plt.plot(dt_, dev_dt[:, 1], marker='o', label='pumpA=3.831')
                    plt.ylabel(r'$error(\bar I_\mathrm{E})$')
                    plt.xlabel('dt')
                    plt.grid()
                    plt.legend()
                    plt.show()
