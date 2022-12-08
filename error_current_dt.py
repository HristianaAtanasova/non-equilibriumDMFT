import argparse
import toml
import zipfile
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import os
import electric_field

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

def calculate_current(dt_, tmax, F):
    av_I = np.zeros(len(F))
    dev_dt = np.zeros((len(dt_),len(dt_)), float)

    # calculate current
    for dt_i in range(len(dt_)):
        dt = dt_[dt_i]
        t  = np.arange(0, tmax, dt)
    
        # loaded = np.load(ferm_Delta_E_name.format(dt, wC, T, tmax))
        # ferm_Delta_E[T_] = loaded['Delta']
        # loaded = np.load(ferm_Delta_name.format(dt, wC, T, tmax))
        # ferm_Delta[T_] = loaded['Delta']
    
        Path = os.getcwd() + '/..'
        for F_ in range(len(F)):
            F = F[F_]

            path = Path.format()
            parser = argparse.ArgumentParser(description = "run dmft")
            parser.add_argument("--params",   default = path + "/run.toml")
            args = parser.parse_args()
    
            with open(args.params, "r") as f:
                params = toml.load(f)
    
            params.update(vars(args))
    
            # load Greens functions
            Green_path = path + '/Green_...npz'
            loaded = np.load(Green_path.format(..))
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

def main():
    parser = argparse.ArgumentParser(description = "calculate current")
    parser.add_argument("--params",   default = "run.toml")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))

    # set up parameters
    dt_ = [0.1, 0.05, 0.02, 0.01, 005]
    tmax = 10
    F = [0.0, 2.0]
    
    av_I, dev_dt = calculate_current(dt_, tmax, F)
    
if __name__ == "__main__":
    main()
