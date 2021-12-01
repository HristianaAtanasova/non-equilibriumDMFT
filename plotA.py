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
# from sympy import Heaviside

def fermi_function(w, mu, beta):
    return 1 / (1 + np.exp(beta * (w - mu)))

def heaviside(x):
    if x == 0:
        return 0.5
    return 0 if x < 0 else 1

#Path = os.getcwd() + '/U{}_dt{}_tmax{}'
Path = os.getcwd() 

# set up parameters
#path = Path.format(U, dt, tmax)
path = Path
parser = argparse.ArgumentParser(description = "run dmft")
parser.add_argument("--params",   default = path + "/run.toml")
args = parser.parse_args()

with open(args.params, "r") as f:
    params = toml.load(f)

params.update(vars(args))

U = params['U']
T = params['T']
dt = params['dt'] 
tmax = params['tmax']
pumpOmega = params['pumpOmega']

t  = np.arange(0, tmax, dt)
msg = 'Executing U = {} | tmax = {}'
print(msg.format(U, tmax))

# load Greens functions
#Green_path = path + '/Green_T={}.npz'
Green_path = 'Green_T={}.npz'
loaded = np.load(Green_path.format(T))
t_ = loaded['t']
dt = params['dt']
Green = loaded['Green']

# fft parameters
dt = dt
Cut_w = np.pi / dt
dw = dt
Cut_t = np.pi / dw

ft = np.arange(-Cut_t, Cut_t, dt)
fw = np.arange(-Cut_w, Cut_w, dw)

N_w = len(fw)
N_t = len(ft)

# convert to plotting time and frequency domain
wmin = -10.0
wmax = 10.0
w = np.arange(wmin, wmax, dw)
w_start = int((N_w/2 + int(wmin/dw)))
w_end = int((N_w/2 + int(wmax/dw)))

tmin = -tmax
t_double = np.arange(tmin, tmax, dt)
t_start = int((N_t/2 + int(tmin/dt)))
t_end = int((N_t/2 + int(tmax/dt)))

G = np.zeros((4, len(t_double)), complex)  # first second index: Gles | Ggtr | Gret | Gadv
Heaviside_ret = np.zeros(len(t_double))
Heaviside_adv = np.zeros(len(t_double))
for t1 in range(len(t_double)):
    Heaviside_ret[t1] = heaviside(t_double[t1])
    Heaviside_adv[-t1] = heaviside(t_double[t1])
    Heaviside_adv[0] = heaviside(-t_double[0])

t_lower = 0 
t_0 = int(len(t_double) / 2)
t_upper = len(t_double)

G[0, t_lower:t_0] = 1j * (Green[1, 0, 0:len(t), len(t)-1] + Green[1, 1, 0:len(t), len(t)-1]) 
G[0, t_0:t_upper] = 1j * (Green[1, 0, len(t)-1,  0:len(t)][::-1] + Green[1, 1, len(t)-1,  0:len(t)][::-1])

G[1, t_lower:t_0] = 1j * (Green[0, 0, 0:len(t), len(t)-1] + Green[0, 1, 0:len(t), len(t)-1]) 
G[1, t_0:t_upper] = 1j * (Green[0, 0, len(t)-1,  0:len(t)][::-1] + Green[0, 1, len(t)-1,  0:len(t)][::-1])

G[0] = G[0] # Gles(t1,t2) = -np.conj(Gles(t2,t1))
G[1] = np.conj(G[1]) # Ggtr(t1,t2) = -(-np.conj(Ggtr(t2,t1)))

plt.plot(t_double, np.real(G[0]), '-', t_double, np.imag(G[0]), '--', label='gtr')
plt.plot(t_double, np.real(G[1]), '-', t_double, np.imag(G[1]), '--', label='les')
plt.legend()
plt.savefig('G_time.pdf')
plt.close()

# Gret = -Theta(t)*(Ggtr - Gles)
G[2] = -Heaviside_ret*(G[1] - G[0])

# Gadv = -Theta(-t)*(Ggtr - Gles)
G[3] = -Heaviside_adv*(G[1] - G[0])

# # save energy current as .out file
# np.savetxt(path +'/Gles.out', G[0].view(float))
# np.savetxt(path +'/Ggtr.out', G[1].view(float))
# np.savetxt(path +'/Gret.out', G[2].view(float))
# np.savetxt(path +'/Gadv.out', G[3].view(float))

# fft of Greens functions
G_N = np.zeros((4, N_t), complex) #  Gles | Ggtr | Gret | Gadv
G_N[:, t_start:t_end] = G

fG = ifftshift(fft(fftshift(G_N))) * dt/(2*np.pi)
A = (fG[2]+fG[3]) / 2
f = np.imag(fG[0]) / (2*np.imag(A))

plt.plot(fw[w_start:w_end], np.imag(A[w_start:w_end]))
# plt.plot(fw[w_start:w_end], np.imag(fG[2][w_start:w_end]))
# plt.plot(fw[w_start:w_end], f[w_start:w_end])
# plt.plot(w, fermi_function(w, 0, 1.0), color='red', label='fermi_dirac')
# plt.legend()
# plt.grid()
# plt.ylim(-0.5,1.5)
# plt.show()
plt.savefig('A.pdf')
