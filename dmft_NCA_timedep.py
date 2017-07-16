from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime
from scipy.integrate import quad
from scipy.signal import hilbert
from numpy.linalg import inv
from numpy.linalg import solve

# set parameters
T = 1
beta = 1 / T
mu = 0
wC = 10
v_0 = 1
v = 10
treshold = 1e-6
tmax = 2
dt = 0.01
t = np.arange(0, tmax, dt)

########################################################################################################################
''' Calculate initial time-domain Hybridization function for given Density of States '''

dw = 0.01
wDOS = np.arange(-2 * v_0, 2 * v_0, dw)
Cut = np.pi / dt
w = np.arange(-Cut, Cut, dw)
fft_tmax = np.pi / dw
fft_tmin = -np.pi / dw
fft_dt = np.pi / Cut
fft_time = np.arange(fft_tmin, fft_tmax, fft_dt)


# fermi function
def fermi_function(w):
    return 1 / (1 + np.exp(beta * (w - mu)))


# flat band with soft cutoff
def A(w):
    return 1 / ((np.exp(v * (w - wC)) + 1) * (np.exp(-v * (w + wC)) + 1))


# semicircular density of states for bethe lattice
def semicircularDos(w):
    return 1 / (2 * np.pi * v_0 ** 2) * np.sqrt(4 * v_0 ** 2 - w ** 2)


# window function padded with zeros for semicircular DOS
N = int(2*Cut/dw)
a = int(N/2+2*v_0/dw)
b = int(N/2-2*v_0/dw)
DOS = np.zeros(N+1)
DOS[b:a] = semicircularDos(wDOS)
#DOS = hilbert(DOS)

# frequency-domain Hybridization function
# Hyb_les = DOS * fermi_function(w)
# Hyb_gtr = DOS * (1 - fermi_function(w))

Hyb_les = A(w) * fermi_function(w)
Hyb_gtr = A(w) * (1 - fermi_function(w))

fDelta_les = np.conj(ifftshift(fft(fftshift(Hyb_les)))) * dw/np.pi
fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi

# get real times from fft_times
Delta_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down
for t_ in range(len(t)):
    # Delta_init[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
    # Delta_init[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
    Delta_init[0, :, t_] = fDelta_gtr[int(N / 2) + t_]
    Delta_init[1, :, t_] = fDelta_les[int(N / 2) + t_]

# time-dependent hopping due to turning on of electric field
t_turn_on = 0.5
t0 = 0.1
F_0 = 0

# def ramp(x):
#     if t_turn_on <= x <= t0+t_turn_on:
#         return 1/2 - 3/4 * np.cos(np.pi*(x-t_turn_on)/t0) + 1/4 * (np.cos(np.pi*(x-t_turn_on)/t0))**3
#     elif x < t_turn_on:
#         return 0
#     else:
#         return 1

def ramp(x):
    if t_turn_on <= x <= t0+t_turn_on:
        return U + (Uc-U)*(1/2 - 3/4 * np.cos(np.pi*(x-t_turn_on)/t0) + 1/4 * (np.cos(np.pi*(x-t_turn_on)/t0))**3)
    elif x < t_turn_on:
        return U
    return Uc

def integrateF(t):
    return F_0*quad(ramp,0,t)[0]

# F = []
# for i in t:
#     F.append(F_0*ramp(i))
#
# A = []
# for i in t:
#     A.append(-integrateF(i))
#
# v_t = v_0 * (np.cos(A) + 1j*np.sin(A))

########################################################################################################################
''' Impurity solver based on NCA '''


########## Define functions for the impurity solver ##########

def check(a):
    for i in np.nditer(a):
        if abs(i) > treshold:
            return True
    return False


def trapezConv(a, b):
    return dt * (fftconvolve(a, b)[:len(a)] - 0.5 * a[:] * b[0] - 0.5 * a[0] * b[:])


def tdiff(D, t2, t1):
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])


def delta(f, i):
    return 1 if f == i else 0


def weights(A):
    if len(A) == 1:
        return 0
    if len(A) == 2:
        return 0.5*(A[0]+A[1])
    if len(A) == 3:
        return (A[0]+4*A[1]+A[2])/3
    return np.sum(A) - (A[0]+A[-1])*2/3 + (A[1]+A[-2])*7/24 - (A[2]+A[-3])*1/6 + (A[3]+A[-4])*1/24


def w(A):
    if len(A) == 1:
        return 0
    if len(A) == 2:
        return (1/2)**2
    if len(A) == 3:
        return (1/3)**2
    if len(A) == 4:
        return (3/8)**2
    return (1/3)**2


# Self energy for vertex functions
def SelfEnergy(K, DeltaMatrix, Sigma):
    for f in range(4):
        for j in range(4):
            Sigma[f] += K[j] * DeltaMatrix[j, f]


# fill in initial DeltaMatrix for two contour times with initial Delta for time-differences
def initDeltaMatrix(Delta):
    for t1 in range(len(t)):
        for t2 in range(len(t)):
            DeltaMatrix[0, 1, t1, t2] = tdiff(Delta[1, 0], t1, t2)
            DeltaMatrix[0, 2, t1, t2] = tdiff(Delta[1, 1], t1, t2)

            DeltaMatrix[1, 0, t1, t2] = tdiff(Delta[0, 0], t1, t2)
            DeltaMatrix[1, 3, t1, t2] = tdiff(Delta[1, 1], t1, t2)

            DeltaMatrix[2, 0, t1, t2] = tdiff(Delta[0, 1], t1, t2)
            DeltaMatrix[2, 3, t1, t2] = tdiff(Delta[1, 0], t1, t2)

            DeltaMatrix[3, 1, t1, t2] = tdiff(Delta[0, 1], t1, t2)
            DeltaMatrix[3, 2, t1, t2] = tdiff(Delta[0, 0], t1, t2)


def fillDeltaMatrix(Delta):
    # fill in DeltaMatrix for all times | first index is gtr/les | second is spin up/spin down
    # initial dot state |0> can only go to |up> or |down> with a Delta_les
    DeltaMatrix[0, 1] = Delta[1, 0]
    DeltaMatrix[0, 2] = Delta[1, 1]

    # initial dot state |up> can only go to |0> or |up,down>
    DeltaMatrix[1, 0] = Delta[0, 0]
    DeltaMatrix[1, 3] = Delta[1, 1]

    # initial dot state |down> can only go to |0> or |up,down>
    DeltaMatrix[2, 0] = Delta[0, 1]
    DeltaMatrix[2, 3] = Delta[1, 0]

    # initial dot state |up,down> can only go to |up> or |down> with a Delta_gtr
    DeltaMatrix[3, 1] = Delta[0, 1]
    DeltaMatrix[3, 2] = Delta[0, 0]


########## Impurity solver computes two-times correlation functions Green for a given hybridization Delta and interaction U ##########

def Solver(DeltaMatrix, U, init):

    ########## Computation of bold propagators on seperate branches of the contour ##########

    # set energy states
    # epsilon = -U / 2.0
    # E = [0*epsilon, epsilon, epsilon, 2 * epsilon + U]

    # initialize bare propagators
    start = datetime.now()

    E = np.zeros((4, len(t)), int)
    G_0 = np.zeros((4, len(t), len(t)), complex)  # g_0 as a two-times Matrix for convolution integrals
    G = np.zeros((4, len(t), len(t)), complex)  # indices are initial state, contour times located on the same branch t_n, t_m (propagation from t_m to t_n)
    Sigma = np.zeros((4, len(t), len(t)), complex)

    E[1] = -U_/2
    E[2] = -U_/2

    # Computation of bare propagators g_0
    for i in range(4):
        for t1 in range(len(t)):
            for t2 in range(t1+1):
                G_0[i, t1, t2] = np.exp(-1j * E[i, t1-t2] * t[t1-t2])

    plt.plot(t, np.real(G_0[1, len(t) - 1]), 'r--', t, np.imag(G_0[1, len(t) - 1]), 'b--')
    plt.show()

    # main loop over every pair of times t_n and t_m (located on the same contour-branch), where t_m is the smaller contour time
    # take integral over t1 outside the t_m loop
    for t_n in range(len(t)):
        sum_t1 = np.zeros((4, t_n+1), complex)
        for t_m in range(t_n, -1, -1):

            sum_t2 = np.zeros(4, complex)
            for i in range(4):
                sum_t2[i] = dt**2 * weights(G[i, t_m:t_n+1, t_m]*sum_t1[i, t_m:])

            # Dyson equation for time (t_m, t_n)
            G[:, t_n, t_m] = (G_0[:, t_n, t_m] - sum_t2) / (1 + dt ** 2 * G_0[:, t_n, t_n] * Sigma[:, t_n, t_n] * w(G[i, t_m:t_n+1, t_m]*sum_t1[i, t_m:]))

            # Compute self-energy for time (t_m, t_n)
            Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrix[:, :, t_n, t_m], 1)

            for i in range(4):
                sum_t1[i, t_m] = weights(Sigma[i, t_m:t_n+1, t_m] * G_0[i, t_n, t_m:t_n+1])  # sum[:, t2=t_m]

    for t_n in range(len(t)):
        for t_m in range(t_n+1, len(t), +1):
                G[:, t_n, t_m] = np.conj(G[:, t_m, t_n])

    for i in range(4):
        plt.plot(t, np.real(G[i, len(t)-1, ::-1]), 'r--', t, np.imag(G[i, len(t)-1, ::-1]), 'b--')
        plt.show()

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Finished calculation of bold propagators after', datetime.now() - start)
    print('                                                                                                                               ')

    ########## Computation of Vertex Functions including hybridization lines between the upper and lower branch ##########

    start = datetime.now()
    K = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial state | contour times on lower and upper branch
    K_0 = np.zeros((4, 4, len(t), len(t)), complex)

    # for every initial state i there is a set of 4 coupled equations for K
    # for i in range(4):
    i = init
    start_i = datetime.now()
    for f in range(4):
        K_0[i, f] = delta(i, f) * np.conj(G[i, :, None, 0]) * G[i, None, :, 0]

    Sigma = np.zeros((4, len(t), len(t)), complex)  # indices are final state | lower and upper branch time

    for t_n in range(len(t)):
        sum_t1 = np.zeros((4, len(t)), complex)
        for t_m in range(len(t)):

            sum_t2 = np.zeros(4, complex)
            M = np.eye(4, dtype=complex)

            for f in range(4):
                sum_t2[f] = dt**2 * np.trapz(G[f, t_m, :t_m+1] * sum_t1[f, :t_m+1])
                # sum_t2[f] = dt ** 2 * weights(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

                M[f] += dt**2*np.sum(DeltaMatrix[f, :, t_n, t_m]*np.conj(G[:, t_n, t_n])*G[:, t_m, t_m], 0) * 1/4
                # M[f] -= dt**2*np.sum(DeltaMatrix[f, :, t_n, t_m]*np.conj(G[:, t_n, t_n])*G[:, t_m, t_m], 0) * w(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

            # Dyson equation for time (t_m, t_n)
            K[i, :, t_n, t_m] = np.linalg.solve(M, K_0[i, :, t_n, t_m] + sum_t2)

            # Compute self-energy for time (t_m, t_n)
            SelfEnergy(K[i, :, t_n, t_m], DeltaMatrix[:, :, t_n, t_m], Sigma[:, t_n, t_m])

            for f in range(4):
                sum_t1[f, t_m] = np.trapz(Sigma[f, :t_n+1, t_m] * np.conj(G[f, t_n, :t_n+1]))  # t_m = t_2
                # sum_t1[f, t_m] = weights(Sigma[f, :t_n+1, t_m] * np.conj(G[f, t_n, :t_n+1]))  # t_m = t_2

    print('Finished calculation of K for initial state', i, 'after', datetime.now() - start_i)
    err = np.abs(1 - np.abs(np.sum(K[i, :, len(t)-1, len(t)-1], 0)))
    print('Error for inital state =', i, 'is', err)
    print('                                                                                                                               ')

    # output
    file = 'K_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}_f={}_test.out'
    for f in range(4):
        np.savetxt(file.format(U, T, tmax, dt, t_turn_on, F_0, i, f), K[i, f].view(float), delimiter=' ')

    # plt.plot(t, np.real(K[i, 1, :, len(t) - 1]), 'r--', t, np.imag(K[i, 1, :, len(t) - 1]), 'b--')
    # plt.plot(t, np.real(K[i, 1, len(t) - 1]), 'y--', t, np.imag(K[i, 1, len(t) - 1]), 'k--')
    # plt.grid()
    # plt.show()

    ########## Computation of two-times Green's functions ##########
    for t1 in range(len(t)):
        for t_1 in range(len(t)):
            Green[0, 0, i, t1, t_1] = (K[i, 0, t1, t_1] * G[1, t1, t_1] + K[i, 2, t1, t_1] * G[3, t1, t_1])
            Green[1, 0, i, t1, t_1] = (K[i, 1, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[2, t1, t_1])
            Green[0, 1, i, t1, t_1] = (K[i, 0, t1, t_1] * G[2, t1, t_1] + K[i, 1, t1, t_1] * G[3, t1, t_1])
            Green[1, 1, i, t1, t_1] = (K[i, 2, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[1, t1, t_1])

    # for t_n in range(len(t)):
    #     for t_m in range(t_n + 1, len(t), +1):
    #         Green[:, :, :, t_n, t_m] = np.conj(Green[:, :, :, t_m, t_n])

    # plt.plot(t, np.real(Green[1, 0, 1, len(t) - 1]), 'y--', t, np.imag(Green[1, 0, 1, len(t) - 1]), 'k--')
    # plt.plot(t, np.real(Green[1, 0, 1, :, len(t) - 1]), 'r--', t, np.imag(Green[1, 0, 1, :, len(t) - 1]), 'b--')
    # plt.plot(t, np.real(Green[0, 0, 1, len(t) - 1]), 'r--', t, np.imag(Green[0, 0, 1, len(t) - 1]), 'b--')
    # plt.grid()
    # plt.show()

    # plt.plot(t, np.real(Green[0, 0, 1, len(t)-1]), 'y--', t, np.imag(Green[0, 0, 1, len(t)-1]), 'k--')
    # plt.plot(t, np.real(Green[1, 1, 1, len(t)-1]), 'r--', t, np.imag(Green[1, 1, 1, len(t)-1]), 'b--')
    # plt.grid()
    # plt.show()

    # output
    gtr_up = 'gtr_up_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}_test.out'
    les_up = 'les_up_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}_test.out'
    np.savetxt(gtr_up.format(U, T, tmax, dt, t_turn_on, F_0, i), Green[0, 0, i].view(float), delimiter=' ')
    np.savetxt(les_up.format(U, T, tmax, dt, t_turn_on, F_0, i), Green[1, 0, i].view(float), delimiter=' ')

    gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}_test.out'
    les_down = 'les_down_U={}_T={}_t={}_dt={}_turn={}_F0={}_i={}_test.out'
    np.savetxt(gtr_down.format(U, T, tmax, dt, t_turn_on, F_0, i), Green[0, 1, i].view(float), delimiter=' ')
    np.savetxt(les_down.format(U, T, tmax, dt, t_turn_on, F_0, i), Green[1, 1, i].view(float), delimiter=' ')

    print('1 - (Green_gtr + Green_les) for Spin Up site', i, 'is', 1 - np.real(Green[0, 0, i, len(t)-1, len(t)-1] + Green[1, 0, i, len(t)-1, len(t)-1]))
    print('1 - (Green_gtr + Green_les) for Spin Down site', i, 'is', 1 - np.real(Green[0, 1, i, len(t)-1, len(t)-1] + Green[1, 1, i, len(t)-1, len(t)-1]))

    print('                                                                                                                               ')

    print('Population for Spin Up les on site', i, 'is', Green[1, 0, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Down les on site', i, 'is', Green[1, 1, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Up gtr on site', i, 'is', Green[0, 0, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Down gtr on site', i, 'is', Green[0, 1, i, len(t) - 1, len(t) - 1])

    print('                                                                                                                               ')


########################################################################################################################
''' Main part starts here '''

Umax = 4
Umin = 3

######### perform loop over U #########
for U in np.arange(Umin, Umax, 2):

    U_ = np.zeros(len(t), float)
    Uc = 3
    # t0 = int(1/(2*dt))
    # U_[0:t0] = U
    # U_[t0:len(t)+1] = 8

    # smooth tuning of U
    for t_ in range(len(t)):
        U_[t_] = ramp(t[t_])

    plt.plot(t, U_)
    plt.show()

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Starting DMFT loop for U =', U, '| Temperature =', T, '| time =', tmax, '| dt =', dt, '| F0=', F_0, '| turn on =', t_turn_on)
    start = datetime.now()
    print('                                                                                                                               ')

    Green = np.zeros((2, 2, 4, len(t), len(t)), complex)  # Greens function with indices gtr/les | spin up/spin down | initial state | lower and upper branch time
    Green_old = np.zeros((2, 2, 4, len(t), len(t)), complex)
    DeltaMatrix = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial and final states

    # first DMFT loop with initial guess for Delta
    initDeltaMatrix(Delta_init)
    Solver(DeltaMatrix, U, 1)
    # Solver(DeltaMatrix, U, 2)
    # Delta = np.zeros((2, 2, 4, len(t), len(t)), complex)  # after the initial guess Delta becomes a two times function --> input into DeltaMatrix; indices are gtr/les | spin up/spin down | initial state
    #
    # counter = 0
    # while np.amax(np.abs(Green_old - Green)) > 0.001:
    #     counter += 1
    #
    #     # Delta[:] = v_0 * np.sum(Green[:, :, :], 2)/4 * v_0
    #
    #     Delta[:] = v_0 * Green[:, :, :] * v_0
    #
    #     Green_old[:] = Green
    #
    #     fillDeltaMatrix(Delta[:, :, 1])
    #     Solver(DeltaMatrix, U, 2)
    #
    #     fillDeltaMatrix(Delta[:, :, 2])
    #     Solver(DeltaMatrix, U, 1)
    #
    #     Diff = np.amax(np.abs(Green_old - Green))
    #     print('for U = ', U, ' and iteration Nr. ', counter, ' the Difference is ', Diff, ' after a calculation time ', datetime.now() - start)
    #     print('                                                                                                                               ')
    #
    #
    # print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    # print('Computation of Greens functions for U =', U, '| Temperature =', T, '| time =', tmax, '| dt =', dt,
    #       '| F0=', F_0, '| turn on =', t_turn_on, 'finished after', counter,
    #       'iterations and', datetime.now() - start, 'seconds.')
    # print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

