#!/usr/bin/env python
from figures import *

from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import argparse
import toml
import zipfile
import numpy as np
import os

Path = os.getcwd() + '/U5.0_probeA{}_probeOmega{}_pumpA{}_pumpOmega{}/T{}'

pumpOmega_ = [3.0]
# pumpOmega_ = [3.0]
probeOmega_ = [1.25]
probeA_ = [0.0]
pumpA_ = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
# pumpA_ = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
Temperature = [0.06, 1.0]

for pumpOmega in pumpOmega_:
    for probeOmega in probeOmega_:
        for probeA in probeA_:
            fig1 = create_figure()
            axes = create_vertical_split(fig1,
                            xlabel=(r'$\mathrm{\omega}$', r'$\mathrm{\omega}$'),
                            ylabel=(r'$\mathrm{F(\omega)}$',r'$\mathrm{F(\omega)}$'),
                            palette=('viridis', 'viridis'),
                            numcolors=(7, 7))
            for i, ax in enumerate(axes):
                T = Temperature[i]
                for pumpA in pumpA_:
                    msg = 'Executing pumpOmega={} | pumpA = {} | probeOmega={} | probeA = {} | T = {}'
                    print(msg.format(pumpOmega, pumpA, probeOmega, probeA, T))

                    if pumpOmega == 0.0:
                        pumpOmega = pumpOmega_[1]
                        path = Path.format(probeA, probeOmega, 0.0, pumpOmega, T)
                    else:
                        path = Path.format(probeA, probeOmega, pumpA, pumpOmega, T)
                    parser = argparse.ArgumentParser(description = "run dmft")
                    parser.add_argument("--params",   default = path + "/run.toml")
                    args = parser.parse_args()

                    with open(args.params, "r") as f:
                        params = toml.load(f)

                    params.update(vars(args))

                    Greensfunct = path + '/Green_T={}.npz'
                    loaded = np.load(Greensfunct.format(T))
                    t = loaded['t']
                    Green = loaded['Green']

                    dt = t[1]-t[0]
                    Cut = np.pi/dt
                    N = int(Cut/dt)
                    ft = np.arange(0, Cut, dt)
                    wmax = np.pi/dt
                    dw = 2*np.pi/Cut
                    fw = np.arange(-wmax, wmax, dw)
                    w = np.arange(-8, 8, dw)

                    period = 2*np.pi/params['pumpOmega']

                    dt_av = period/10
                    t_period = np.arange(params['tmax']-period,params['tmax'],dt_av)
                    f = np.zeros((len(t_period),len(w)), complex)

                    for t_av_ in range(len(t_period)):
                        tmax = t_period[t_av_]
                        t = np.arange(0,tmax,dt)
                        Gles = np.zeros((2,len(t)), complex)
                        Ggtr = np.zeros((2,len(t)), complex)
                        Gles = 1j*np.sum(Green[1, :, len(t)-1, len(t)-1:0:-1],0)/2
                        Ggtr = -1j*np.sum(Green[0, :, len(t)-1:0:-1, len(t)-1],0)/2

                        Gles_ = np.zeros(N, complex)
                        fGles = np.zeros(N, complex)
                        Gles_[0:int(len(t)-1)] = (Gles)

                        Ggtr_ = np.zeros(N, complex)
                        fGgtr = np.zeros(N, complex)
                        Ggtr_[0:int(len(t)-1)] = (Ggtr)

                        Gadv = np.zeros(N, complex)
                        fGadv = np.zeros(N, complex)
                        Gadv[0:int(len(t)-1)] = (Gles - Ggtr)

                        fGles = fftshift(fft(Gles_)) * dt / np.pi
                        fGgtr = fftshift(fft(Ggtr_)) * dt / np.pi
                        fGadv = fftshift(fft(Gadv)) * dt / np.pi
                        a = int((N-len(w))/2)
                        b = int((N+len(w))/2)

                        f[t_av_] = np.imag(fGles[a:b])/np.imag(fGadv[a:b])

                    # plot fermi funtion at temperature T
                    if pumpA == pumpA_[0]:
                        ax.plot(w, 1/(np.exp(w/T)+1), color='red', linestyle='--', label='$f(\omega)$')

                    av_f = np.sum(f,0)/len(t_period)
                    ax.plot(w, np.real(av_f), label='$A = {}$'.format(pumpA))

                    title = '$\mathrm{\omega_\mathrm{pump}}$' + '$={}$'.format(pumpOmega)+ ',' + '$\mathrm{T}$' + '$={}$'.format(T)
                    ax.set_title(title, fontsize='x-small', position=(0.7, 0.8))

            # Clean up and save:
            title = '2_split_distribution_function_pumpOmega={}.pdf'
            finalize_and_save(fig1, title.format(pumpOmega))
