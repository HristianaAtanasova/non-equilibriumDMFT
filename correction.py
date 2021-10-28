#!/usr/bin/env python
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import argparse
import toml
import zipfile
import numpy as np
import os
import scipy.optimize as opt
import scipy.special as special
import matplotlib.pyplot as plt
from sympy import Heaviside

def Heaviside(x):
    return (np.sign(x) + 1)

def Exp(t, params):
    a = params[0]
    b = params[1]
    return a*np.exp(-b*t)

def Exp_ret(t, params):
    a = params[0]
    b = params[1]
    return Heaviside(t)*a*np.exp(-b*t)

def Exp_adv(t, params):
    a = params[0]
    b = params[1]
    return Heaviside(-t)*a*np.exp(b*t)

def FExp_ret(omega):
    return 1/(1+1j*omega)

def FExp_adv(omega):
    return 1/(1-1j*omega)

def sharp_feature(t, w, N_t, ft, t_start, t_end, N_w, fw, w_start, w_end, G_ret, G_adv, params):
    dt = t[1]-t[0]
    f = np.zeros((2, len(t)), complex)
    f[0] = np.imag(G_ret)
    f[1] = np.imag(G_adv)

    f_N = np.zeros((2,N_t),complex)
    f_N[0, t_start:t_end] = G_ret
    f_N[1, t_start:t_end] = G_adv

    fft_f = np.zeros((2,N_w), complex)
    fft_f = ifftshift(fft(fftshift(f_N))) * dt/(2*np.pi)

    fit = np.zeros((2,len(t)), complex)
    fit[0] = Exp_ret(t, params)
    fit[1] = Exp_adv(t, params)

    normalization_ret = f[0,int(len(t)/2)]/fit[0,int(len(t)/2)]
    normalization_adv = f[1,int(len(t)/2)-1]/fit[1, int(len(t)/2)-1]
    fit[0] = normalization_ret*fit[0]
    fit[1] = normalization_adv*fit[1]

    FT_fit = np.zeros((2,len(w)), complex)
    FT_fit[0] = normalization_ret*FExp_ret(w)
    FT_fit[1] = normalization_adv*FExp_adv(w)

    g_N = np.zeros((2,N_t),complex)
    g_N[:, t_start:t_end] = f-fit

    fg = np.zeros((2,N_w), complex)
    fg = ifftshift(fft(fftshift(g_N))) * dt/2

    # FFT of Exponential
    Exp_N = np.zeros((2,N_t),complex)
    Exp_N[:, t_start:t_end] = fit
    fExp = np.zeros((2,N_w), complex)
    fExp = ifftshift(fft(fftshift(Exp_N))) * dt/2 # factor of 2 comes from taking only positive times

    corr_A = np.zeros((2,len(w)), complex)
    corr_A = 1j*(fg[:,w_start:w_end]+FT_fit)/np.pi

    # plot exponential fit
    # plt.plot(t, f[0], color = 'red', label='imag_Gret(t)')
    # # plt.plot(t, fit[0], color = 'black', label='fit(t)')
    # plt.plot(t, f[0]-fit[0], '--', color = 'green', label='g=imag_Gret(t)-fit(t)')
    # #
    # plt.plot(t, f[1], color = 'blue', label='imag_Gadv(t)')
    # # plt.plot(t, fit[1], color = 'black', label='fit(t)')
    # plt.plot(t, f[1]-fit[1], '--', color = 'black', label='g=imag_Gret(t)-fit(t)')

    # test if fft and analytical FT are the same
    # plt.plot(w, np.imag(FT_fit[1]), color = 'blue', label='imag_Exp(omega)')
    # plt.plot(w, np.real(FT_fit[1]), color = 'red', label='real_Exp(omega)')
    # plt.plot(w, np.real(fExp[1,w_start:w_end]), '--', color = 'red', label='fft_Exp(omega)')
    # plt.plot(w, np.imag(fExp[1,w_start:w_end]), '--', color = 'blue', label='fft_Exp(omega)')

    # frequency domain
    # plt.plot(w, np.real(fft_f[0, w_start:w_end]), color='red', label='real_Gret(omega)')
    # plt.plot(w, np.real(corr_A[0]), '--', color = 'red', label='real_g+Exp(omega)')
    # plt.plot(w, np.imag(fft_f[0, w_start:w_end]), color='blue', label='imag_Gret(omega)')
    # plt.plot(w, np.imag(corr_A[0]), '--', color = 'blue', label='imag_g+Exp(omega)')

    # plt.plot(w, np.real(fft_f[1, w_start:w_end]), color='red', label='real_Gadv(omega)')
    # plt.plot(w, np.real(corr_A[1]), '--', color = 'red', label='real_g+Exp(omega)')
    # plt.plot(w, np.imag(fft_f[1, w_start:w_end]), color='blue', label='imag_Gadv(omega)')
    # plt.plot(w, np.imag(corr_A[01]), '--', color = 'blue', label='imag_g+Exp(omega)')
    #
    # plt.legend()
    # plt.grid()
    # plt.show()
    return corr_A
