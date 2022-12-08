import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# parameters characterizing the quench profile from U_ to Uconst

t_turn = 3.0
t0 = 1

def quench_U(t_, U, Uconst):
    if t_turn <= t_ <= t0+t_turn:
        return U + (Uconst-U)*(1/2 - 3/4 * np.cos(np.pi*(t_-t_turn)/t0) + 1/4 * (np.cos(np.pi*(t_-t_turn)/t0))**3)
    elif t_ < t_turn:
        return U
    return Uconst

def integrate_U(t1,t2,U,Uconst):
    return quad(quench_U,t2,t1,args=(U,Uconst))[0]

def bare_prop(t, U, Uconst):
    # bare propagator G_0 as a Matrix for convolution integrals
    G_0 = np.zeros((4, len(t), len(t)), complex)
    U_ = np.zeros(len(t), float)

    for t_ in range(len(t)):
        U_[t_] = quench_U(t[t_], U, Uconst)

    # Computation of bare propagators G_0
    if U == Uconst:
        E = np.zeros((4, len(t)), float)

        E[1, :] = -U/2.0
        E[2, :] = -U/2.0

        for i in range(4):
            for t1 in range(len(t)):
                for t2 in range(t1+1):
                    G_0[i, t1, t2] = np.exp(-1j * E[i, t1-t2] * t[t1-t2])

        for t1 in range(len(t)):
            for t2 in range(t1+1, len(t), +1):
                    G_0[:, t1, t2] = np.conj(G_0[:, t2, t1])
    else:
        for t1 in range(len(t)):
            for t2 in range(len(t)):
                G_0[0, t1, t2] = np.exp(-1j * 0)
                G_0[1, t1, t2] = np.exp(-1j * -integrate_U(t[t1], t[t2],U,Uconst)/2)
                G_0[2, t1, t2] = np.exp(-1j * -integrate_U(t[t1], t[t2],U,Uconst)/2)
                G_0[3, t1, t2] = np.exp(-1j * 0)

    np.savez_compressed('barePropagators', t=t, G_0=G_0)
