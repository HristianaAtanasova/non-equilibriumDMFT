from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

#!/usr/bin/env python
from figures import *
import argparse
import toml
import zipfile
import numpy as np
import os

Path = os.getcwd() + '/U5.0_probeA{}_probeOmega{}_pumpA{}_pumpOmega{}/T{}'

pumpOmega_ = [3.0]
probeOmega_ = [1.25]
probeA_ = [0.0]
pumpA_ = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
Temperature = [0.06, 1.0]

for probeOmega in probeOmega_:
    for probeA in probeA_:
        for pumpOmega in pumpOmega_:
            for T in Temperature:
                fig = create_figure()
                ax = create_single_panel(fig, xlabel=r'$\mathrm{\omega}$', ylabel=r'$\mathrm{A(\omega)}$',
                                        palette=('viridis'), numcolors=(7))
                for pumpA in pumpA_:
                    msg = 'Executing pumpOmega={} | pumpA = {} | T = {}'
                    print(msg.format(pumpOmega, pumpA, T))

                    if pumpOmega == 0.0:
                        pumpOmega = pumpOmega_[1]
                        path = Path.format(probeA, probeOmega, pumpA, 0.0, T)
                    else:
                        path = Path.format(probeA, probeOmega, pumpA, pumpOmega, T)
                    parser = argparse.ArgumentParser(description = "run dmft")
                    parser.add_argument("--params",   default = path + "/run.toml")
                    args = parser.parse_args()

                    with open(args.params, "r") as f:
                        params = toml.load(f)

                    params.update(vars(args))

                    # plot spectral fuction for different temperatures and one pump amplitude
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

                    if pumpA == 0:
                        period = 2*np.pi/params['probeOmega']
                    else:
                        period = 2*np.pi/params['pumpOmega']

                    dt_av = period/10
                    t_period = np.arange(params['tmax']-period,params['tmax'],dt_av)
                    fGadv = np.zeros((len(t_period),2,N), complex)

                    for t_av_ in range(len(t_period)):
                        t_av = t_period[t_av_]
                        t = np.arange(0,t_av,dt)
                        Gles = np.zeros((2,len(t)), complex)
                        Ggtr = np.zeros((2,len(t)), complex)
                        Gles = 1j*Green[1, :, len(t)-1, len(t)-1:0:-1]
                        Ggtr = -1j*Green[0, :, len(t)-1:0:-1, len(t)-1]

                        Gadv = np.zeros((2,N), complex)
                        Gadv[:,0:int(len(t)-1)] = (Gles - Ggtr)
                        fGadv[t_av_] = fftshift(fft(Gadv)) * dt / np.pi

                    a = int((N-len(w))/2)
                    b = int((N+len(w))/2)

                    av_Gadv = np.zeros((int(period/dt_av),N), complex)
                    av_Gadv = (fGadv[:,0] + fGadv[:,1])/2
                    av_Gadv = np.sum(av_Gadv,0)/len(t_period)

                    ax.plot(w, np.imag(av_Gadv[a:b]), linewidth='0.5', label='$A= {}$'.format(pumpA))

                title = '$\mathrm{\omega_\mathrm{pump}}$' + '$={}$'.format(pumpOmega)+ ',' + '$\mathrm{T}$' + '$={}$'.format(T)
                ax.set_title(title, fontsize='x-small', position=(0.5, 0.0))
                # title = '$\mathrm{\omega_\mathrm{probe}}$' + '$={}$'.format(pumpOmega)
                # ax.set_title(title, position=(0.85, 0.8))
                # Clean up and save:
                title = 'spectralFunction_T={}_pumpOmega={}.pdf'
                finalize_and_save(fig, title.format(T, pumpOmega))
