#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import scipy.optimize as opt
import scipy.special as special
import argparse
import toml
import zipfile
import numpy as np
import os
import matplotlib.pyplot as plt
import correction

def fermi_function(w, mu, beta):
    return 1 / (1 + np.exp(beta * (w - mu)))

# def Heaviside(x):
#     return (np.sign(x) + 1)

def heaviside(x):
    if x == 0:
        return 0.5
    return 0 if x < 0 else 1

Path = os.getcwd() + '/../output/U{}_dt{}_pumpA{}_pumpOmega{}/wC{}'

# set up parameters
tmax = 10
U_ = [10.0]
pumpOmega_ = [10.0]
probeOmega_ = [1.25]
# pumpA_ = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.831, 4.5, 5.0, 5.136, 5.5, 6.0, 6.5, 7.015, 7.5, 8.0]
pumpA_ = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.831, 4.5, 5.0, 5.5, 6.0, 6.5, 7.015, 7.5, 8.0]
probeA_ = [0.0]
Temperature_ = [1.0]
tmax = 10
# dt_ = [0.015, 0.0125, 0.01, 0.0075, 0.005]
dt_ = [0.02, 0.015, 0.01, 0.005]
# dt_ = [0.01]
# wC_ = [10, 50]
wC_ = [50]

# calculate current
for U in U_:
    for T_ in range(len(Temperature_)):
        T = Temperature_[T_]
        for probeOmega in probeOmega_:
            for probeA in probeA_:
                for pumpOmega in pumpOmega_:
                    for wC_i in range(len(wC_)):
                        wC = wC_[wC_i]
                        for pumpA_i in range(len(pumpA_)):
                            pumpA = pumpA_[pumpA_i]
                            msg = 'Executing wC = {} | pumpA = {}'
                            print(msg.format(wC, pumpA))

                            wmin = -28.0
                            wmax = 28.0
                            w_final = np.arange(wmin, wmax, 0.01)
                            F = np.zeros((len(dt_), len(w_final)), float)

                            for dt_i in range(len(dt_)):
                                dt = dt_[dt_i]

                                print(dt)

                                path = Path.format(U, dt, pumpA, pumpOmega, wC)
                                parser = argparse.ArgumentParser(description = "run dmft")
                                parser.add_argument("--params",   default = path + "/run.toml")
                                args = parser.parse_args()

                                with open(args.params, "r") as f:
                                    params = toml.load(f)

                                params.update(vars(args))

                                # load Greens functions
                                Green_path = path + '/Green_T={}.npz'
                                loaded = np.load(Green_path.format(T))
                                t_ = loaded['t']
                                dt = params['dt']
                                Green = loaded['Green']

                                # print(np.shape(Green))
                                # fft parameters
                                dt = dt
                                Cut_w = 2*np.pi/dt
                                dw = dt
                                Cut_t = 2*np.pi/dw

                                ft = np.arange(-Cut_t/2, Cut_t/2, dt)
                                fw = np.arange(-Cut_w/2, Cut_w/2, dw)

                                N_w = len(fw)
                                N_t = len(ft)

                                # convert to plotting time and frequency domain
                                wmin = -45.0
                                wmax = 45.0
                                w = np.arange(wmin, wmax, dw)
                                w_start = int((N_w/2+int(wmin/dw)))
                                w_end = int((N_w/2+int(wmax/dw)))

                                tmin = -tmax
                                t = np.arange(tmin, tmax, dt)
                                t_start = int((N_t/2+int(tmin/dt)))
                                t_end = int((N_t/2+int(tmax/dt)))

                                period = 2*np.pi/pumpOmega
                                t_period = np.arange(params['tmax'],params['tmax']-period,-dt)

                                G_p = np.zeros((len(t_period),4,len(t)), complex)           # second index: Gles | Ggtr | Gret | Gadv

                                Heaviside_ret = np.zeros(len(t))
                                Heaviside_adv = np.zeros(len(t))
                                for t_i in range(len(t)):
                                    Heaviside_ret[t_i] = heaviside(t[t_i])
                                    Heaviside_adv[-t_i] = heaviside(t[t_i])
                                    Heaviside_adv[0] = heaviside(-t[0])

                                # period averaged Greens functions
                                steps = 1
                                for i in range(0,len(t_period),steps):
                                    t_p = np.arange(0,t_period[i],dt)

                                    t_lower = int(len(t_)-len(t_p))+1
                                    t_0 = int(len(t)/2)
                                    t_upper = int(len(t_)+len(t_p))

                                    # slicing of Greens funcitons
                                    # Gles
                                    G_p[i,0,t_lower:t_0+1] = 1j*np.conj(Green[1, 0, 0:len(t_p), len(t_p)-1]+Green[1, 1, 0:len(t_p), len(t_p)-1]) #U=10
                                    G_p[i,0,t_0:t_upper] = 1j*np.conj(Green[1, 0, len(t_p)-1,  0:len(t_p)][::-1]+Green[1, 1, len(t_p)-1,  0:len(t_p)][::-1])

                                    # Ggtr
                                    G_p[i,1,t_lower:t_0+1] = 1j*(Green[0, 0, 0:len(t_p), len(t_p)-1]+Green[0, 1, 0:len(t_p), len(t_p)-1]) #U=10
                                    G_p[i,1,t_0:t_upper] = 1j*(Green[0, 0, len(t_p)-1,  0:len(t_p)][::-1]+Green[0, 1, len(t_p)-1,  0:len(t_p)][::-1])

                                    ##################################################################################################################################
                                    G_p[i,0] = -np.conj(G_p[i,0]) # Gles(t1,t2) = -np.conj(Gles(t2,t1))
                                    G_p[i,1] = np.conj(G_p[i,1]) # Ggtr(t1,t2) = -(-np.conj(Ggtr(t2,t1)))

                                    # Gret = -Theta(t)*(Ggtr - Gles)
                                    G_p[i,2] = -Heaviside_ret*(G_p[i,1]-G_p[i,0])

                                    # Gadv = -Theta(-t)*(Ggtr - Gles)
                                    G_p[i,3] = -Heaviside_adv*(G_p[i,1]-G_p[i,0])

                                G = np.sum(G_p,axis=0)/len(t_period)

                                #######################################################################################################################################
                                # fft of Greens functions
                                G_N = np.zeros((4,N_t), complex)                        #  Gles | Ggtr | Gret | Gadv

                                if dt == 0.015:
                                    G_N[:,t_start-1:t_end+1] = G
                                else:
                                    G_N[:,t_start:t_end] = G

                                fG = ifftshift(fft(fftshift(G_N))) * dt/(np.pi)
                                A = (fG[2]+fG[3])/2
                                f = np.imag(fG[0])/(2*np.imag(A))

                                for w_N in range(len(w_final)):
                                    w_val = int((N_w/2+int(w_final[w_N]/dw)))
                                    # print(dt_i, w_N)
                                    F[dt_i, w_N] = (np.imag(fG[0,w_val])/2)/(np.imag(fG[0,w_val])/2-np.imag(fG[1,w_val])/2)

                            Fermi = np.zeros(len(w_final))
                            for w_N in range(len(w_final)):
                                fit = np.polyfit(dt_,F[:, w_N], 2)
                                function_fit = np.poly1d(fit)
                                Fermi[w_N] = function_fit[0]

                            path_to_save = Path.format(U, 0.005, pumpA, pumpOmega, wC)
                            # np.savetxt(path_to_save +'/extrapol_Fermi.out', Fermi)
                            np.savetxt(path_to_save +'/extrapol_Fermi.out', np.c_[w_final,Fermi])

                            # plt.plot(w_final, Fermi)
                            # plt.plot(w, fermi_function(w, 0, 1.0), color='red', label='fermi_dirac')
                            # plt.legend()
                            # plt.grid()
                            # plt.ylim(-0.5,1.5)
                            # plt.show()
