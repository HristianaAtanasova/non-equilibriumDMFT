from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime
from scipy.integrate import quad
from scipy.signal import hilbert
from scipy.special import jv
from numpy.linalg import inv
from numpy.linalg import solve

## set parameters
# physical parameters
T = 1
beta = 1 / T
mu = 0
v_0 = 1
w_0 = 1
gamma = 1
# Amplitude = 3

# numerical parameters
phonon_bath = 1
fermion_bath = 0
lambda_const = 1
counter_term = 1
rho = 6
wC = 10
v = 10
delta_width = 0.1
treshold = 1e-6

# time domain
tmax = 8
dt = 0.005
t = np.arange(0, tmax, dt)

# frequency domain
dw = 0.01
wDOS = np.arange(-2 * v_0, 2 * v_0, dw)
Cut = np.pi / dt
w = np.arange(-Cut, Cut, dw)
fft_tmax = np.pi / dw
fft_tmin = -np.pi / dw
fft_dt = np.pi / Cut
fft_time = np.arange(fft_tmin, fft_tmax, fft_dt)

########################################################################################################################
''' Calculate Hybridization function for the Fermion Bath and the Phonon Bath '''

########## Define functions ###########

def coth(w):
    return (np.exp(w*beta/2)+1.0)/(np.exp(w*beta/2)-1.0)


def phononSpectral(w):
    # return (w-mu)/(rho-mu) / (np.exp(v*(w-rho)) + 1)
    return (1 / w_0) * (np.exp(-((w-w_0)/delta_width)**2) / (np.sqrt(np.pi)*delta_width))  # Delta function
    # return (1/(np.pi*0.1)) * (1 / (1 + ((w-1)/0.1)**2))   # Lorenz distribution


def phononCorrelation(t, w):
    return coth(w) * np.cos(w*t) - 1j * np.sin(w*t)


def phononBath_(t,w):
    return phononSpectral(w) * w * phononCorrelation(t, w)


# fermi distribution
def fermi_function(w):
    return 1 / (1 + np.exp(beta * (w - mu)))


# bose distribution
def bose_function(w):
    return 1 / (np.exp(beta * w) - 1)


# flat band with soft cutoff
def A(w):
    return gamma / ((np.exp(v * (w - wC)) + 1) * (np.exp(-v * (w + wC)) + 1))


# semicircular density of states for bethe lattice
def semicircularDos(w):
    return 1 / (2 * np.pi * v_0 ** 2) * np.sqrt(4 * v_0 ** 2 - w ** 2)

#
# plt.plot(wDOS, phononSpectral(wDOS),'r')
# plt.show()
######## Calculate Hybridization functions ###########

# window function padded with zeros for semicircular DOS
N = int(2*Cut/dw)
a = int(N/2+2*v_0/dw)
b = int(N/2-2*v_0/dw)
DOS = np.zeros(N+1)
DOS[b:a] = semicircularDos(wDOS)
#DOS = hilbert(DOS)

# frequency-domain Hybridization function
Hyb_les = DOS * fermi_function(w)
Hyb_gtr = DOS * (1 - fermi_function(w))

# Hyb_les = A(w) * fermi_function(w)
# Hyb_gtr = A(w) * (1 - fermi_function(w))

# fDelta_les = -1j * ifftshift(fft(fftshift(Hyb_les))) * dw/np.pi
# fDelta_gtr = 1j * ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi

fDelta_les = np.conj(ifftshift(fft(fftshift(Hyb_les)))) * dw/np.pi
fDelta_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi

ffermionBath = ifftshift(fft(fftshift(A(w)*(1-fermi_function(w))))) * dw/np.pi

# get real times from fft_times
Delta_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down
phononBath = np.zeros((len(t)), complex)
fermionBath = np.zeros((len(t)), complex)
fermion_G = np.zeros((len(t)), complex)

for t_ in range(len(t)):

    # Delta_init[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
    # Delta_init[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]

    Delta_init[0, :, t_] = fDelta_gtr[int(N / 2) + t_]
    Delta_init[1, :, t_] = fDelta_les[int(N / 2) + t_]

    # phononBath[t_] = fPhononBath[int(N / 2) + t_]

    phononBath[t_] = (quad(lambda w: np.real(phononBath_(t[t_],w)), -100, -0.01, limit=600)[0] + quad(lambda w: np.real(phononBath_(t[t_],w)), 0.01, 100, limit=600)[0]
                                  + 1j * quad(lambda w: np.imag(phononBath_(t[t_],w)), -100, -0.01, limit=600)[0] + 1j * quad(lambda w: np.imag(phononBath_(t[t_],w)), 0.01, 100, limit=600)[0])

    # phononBath[t_] = (2 / np.pi) * (quad(lambda w: np.real(phononBath_(t[t_], w)), 0, 10, limit=300)[0] + 1j * quad(lambda w: np.imag(phononBath_(t[t_], w)), 0, 10, limit=300)[0])

    fermionBath[t_] = ffermionBath[int(N / 2) + t_]

########################################################################################################################
''' Time-dependent coupling to the phonon Bath lambda | hopping v | interaction U '''

t_turn = 3.0
t0 = 1
F_0 = 0

def tune_Coupling(x):
    return lambda_const / (1 + np.exp(10 * (x - 20)))


def tune_U(x):
    if t_turn <= x <= t0+t_turn:
        return U + (Uc-U)*(1/2 - 3/4 * np.cos(np.pi*(x-t_turn)/t0) + 1/4 * (np.cos(np.pi*(x-t_turn)/t0))**3)
    elif x < t_turn:
        return U
    return Uc


def tune_Hopping(x):
    if t_turn <= x <= t0+t_turn:
        return 1/2 - 3/4 * np.cos(np.pi*(x-t_turn)/t0) + 1/4 * (np.cos(np.pi*(x-t_turn)/t0))**3
    elif x < t_turn:
        return 0
    else:
        return 1


def integrate_F(t):
    return F_0*quad(tune_Hopping,0,t)[0]


def integrate_U(t1,t2):
    return quad(tune_U,t2,t1)[0]

########################################################################################################################
''' Impurity solver based on NCA '''

########## Define functions for impurity solver ##########

def delta(f, i):
    return 1 if f == i else 0


def tdiff(D, t2, t1):
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])


def check(a):
    for i in np.nditer(a):
        if abs(i) > treshold:
            return True
    return False


def trapezConv(a, b):
    return dt * (fftconvolve(a, b)[:len(a)] - 0.5 * a[:] * b[0] - 0.5 * a[0] * b[:])


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
def initMatrix(Delta_init, phononBath, fermionBath, coupling, v_t):
    for t1 in range(len(t)):
        for t2 in range(len(t)):

            Delta[0, 0, t1, t2] = tdiff(Delta_init[0, 0], t1, t2)
            Delta[0, 1, t1, t2] = tdiff(Delta_init[0, 1], t1, t2)
            Delta[1, 0, t1, t2] = tdiff(Delta_init[1, 0], t1, t2)
            Delta[1, 1, t1, t2] = tdiff(Delta_init[1, 1], t1, t2)

            PhononBath[t1, t2] = tdiff(phononBath, t1, t2)

            FermionBath[t1, t2] = tdiff(fermionBath, t1, t2)

            Coupling[t1, t2] = coupling[t1]*coupling[t2]

            V_t[t1, t2] = v_t[t1]*np.conj(v_t[t2])


def fillDeltaMatrixG(Delta):
    # fill in DeltaMatrix for all times | first index is gtr/les | second is spin up/spin down
    # initial dot state |0> can only go to |up> or |down> with a Delta_gtr
    DeltaMatrixG[0, 1] = Delta[1, 0]
    DeltaMatrixG[0, 2] = Delta[1, 1]

    # initial dot state |up> can only go to |0> or |up,down>
    DeltaMatrixG[1, 0] = Delta[0, 0]
    DeltaMatrixG[1, 3] = Delta[1, 1]

    # initial dot state |down> can only go to |0> or |up,down>
    DeltaMatrixG[2, 0] = Delta[0, 1]
    DeltaMatrixG[2, 3] = Delta[1, 0]

    # initial dot state |up,down> can only go to |up> or |down> with a Delta_les
    DeltaMatrixG[3, 1] = Delta[0, 1]
    DeltaMatrixG[3, 2] = Delta[0, 0]


########## Impurity solver computes two-times correlation functions/Greens functions ##########

def Solver(U, init):

    ########## Computation of bold propagators on separate branches of the contour ##########

    # initialize  propagators
    # start = datetime.now()
    G = np.zeros((4, len(t), len(t)), complex)  # indices are initial state, contour times located on the same branch t_n, t_m (propagation from t_m to t_n)
    Sigma = np.zeros((4, len(t), len(t)), complex)

    # main loop over every pair of times t_n and t_m (located on the same contour-branch), where t_m is the smaller contour time
    # take integral over t1 outside the t_m loop
    for t_n in range(len(t)):
        sum_t1 = np.zeros((4, t_n+1), complex)
        for t_m in range(t_n, -1, -1):

            sum_t2 = np.zeros(4, complex)
            for i in range(4):
                sum_t2[i] = dt**2 * weights(G[i, t_m:t_n+1, t_m]*sum_t1[i, t_m:])
                # sum_t2[i] = dt ** 2 * np.trapz(G[i, t_m:t_n + 1, t_m] * sum_t1[i, t_m:])

            # Dyson equation for time (t_m, t_n)
            G[:, t_n, t_m] = (G_0[:, t_n, t_m] - sum_t2) / (1 + dt ** 2 * G_0[:, t_n, t_n] * Sigma[:, t_n, t_n] * w(G[i, t_m:t_n+1, t_m]*sum_t1[i, t_m:]))
            # G[:, t_n, t_m] = (G_0[:, t_n, t_m] - sum_t2) / (1 + dt ** 2 * G_0[:, t_n, t_n] * Sigma[:, t_n, t_n] * 1/4)

            # Compute self-energy for time (t_m, t_n)
            Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrixG[:, :, t_n, t_m], 1)

            if phonon_bath == 1:
                Sigma[0, t_n, t_m] += 2*Coupling[t_n, t_m] * G[0, t_n, t_m] * PhononBath[t_n, t_m]
                Sigma[3, t_n, t_m] += 2*Coupling[t_n, t_m] * G[3, t_n, t_m] * PhononBath[t_n, t_m]

            for i in range(4):
                sum_t1[i, t_m] = weights(Sigma[i, t_m:t_n+1, t_m] * G_0[i, t_n, t_m:t_n+1])  # sum[:, t2=t_m]
                # sum_t1[i, t_m] = np.trapz(Sigma[i, t_m:t_n+1, t_m] * G_0[i, t_n, t_m:t_n+1])  # sum[:, t2=t_m]

    for t_n in range(len(t)):
        for t_m in range(t_n+1, len(t), +1):
                G[:, t_n, t_m] = np.conj(G[:, t_m, t_n])

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Finished calculation of bold propagators after', datetime.now() - start)
    print('                                                                                                                               ')

    ########## Computation of Vertex Functions including hybridization lines between the upper and lower branch ##########

    # start = datetime.now()
    K = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial state | contour times on lower and upper branch
    K_0 = np.zeros((4, 4, len(t), len(t)), complex)
    Sigma = np.zeros((4, len(t), len(t)), complex)  # indices are final state | lower and upper branch time

    # for every initial state i there is a set of 4 coupled equations for K
    # for i in range(4):
    i = init
    for f in range(4):
        K_0[i, f] = delta(i, f) * np.conj(G[i, :, None, 0]) * G[i, None, :, 0]

    for t_n in range(len(t)):
        sum_t1 = np.zeros((4, len(t)), complex)
        for t_m in range(len(t)):

            sum_t2 = np.zeros(4, complex)
            M = np.eye(4, dtype=complex)

            for f in range(4):
                sum_t2[f] = dt**2 * np.trapz(G[f, t_m, :t_m+1] * sum_t1[f, :t_m+1])
                # sum_t2[f] = dt ** 2 * weights(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

                M[f] -= dt**2*np.sum(DeltaMatrixG[f, :, t_n, t_m]*np.conj(G[:, t_n, t_n])*G[:, t_m, t_m], 0) * 1/4
                # M[f] -= dt**2 * np.sum(DeltaMatrix[f, :, t_n, t_m] * np.conj(G[:, t_n, t_n]) * G[:, t_m, t_m], 0) * w(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

            # Dyson equation for time (t_m, t_n)
            K[i, :, t_n, t_m] = np.linalg.solve(M, K_0[i, :, t_n, t_m] + sum_t2)

            # Compute self-energy for time (t_m, t_n)
            SelfEnergy(K[i, :, t_n, t_m], DeltaMatrixG[:, :, t_n, t_m], Sigma[:, t_n, t_m])

           # Add contribution from the phonon Bath
            if phonon_bath == 1:
                Sigma[0, t_n, t_m] += 2*Coupling[t_n, t_m] * K[i, 0, t_n, t_m] * PhononBath[t_n, t_m]
                Sigma[3, t_n, t_m] += 2*Coupling[t_n, t_m] * K[i, 3, t_n, t_m] * PhononBath[t_n, t_m]

            for f in range(4):
                sum_t1[f, t_m] = np.trapz(Sigma[f, :t_n+1, t_m] * np.conj(G[f, t_n, :t_n+1]))  # t_m = t_2
                # sum_t1[f, t_m] = weights(Sigma[f, :t_n+1, t_m] * np.conj(G[f, t_n, :t_n+1]))  # t_m = t_2

    print('Finished calculation of K for initial state', i, 'after', datetime.now() - start)
    err = np.abs(1 - np.abs(np.sum(K[i, :, len(t)-1, len(t)-1], 0)))
    print('Error for inital state =', i, 'is', err)
    print('                                                                                                                               ')

    # output
    # file = 'K_U={}_T={}_t={}_dt={}_A={}_lambda={}_i={}_f={}.out'
    file = 'K_U={}_T={}_t={}_dt={}_A={}_lambda={}_i={}_f={}_ph.out'
    for f in range(4):
        # np.savetxt(file.format(U, T, tmax, dt, Amplitude, gamma, i, f), K[i, f].view(float), delimiter=' ')
        np.savetxt(file.format(U, T, tmax, dt, Amplitude, lambda_const, i, f), K[i, f].view(float), delimiter=' ')

    # plt.plot(t, np.real(K[i,0].diagonal()), 'y--', t, np.real(K[i, 3].diagonal()), 'k--')
    # plt.grid()
    # plt.show()

    ########## Computation of two-times Green's functions ##########
    for t1 in range(len(t)):
        for t_1 in range(len(t)):

            Green[0, 0, i, t1, t_1] = (K[i, 0, t1, t_1] * G[1, t1, t_1] + K[i, 2, t1, t_1] * G[3, t1, t_1])
            Green[1, 0, i, t1, t_1] = (K[i, 1, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[2, t1, t_1])
            Green[0, 1, i, t1, t_1] = (K[i, 0, t1, t_1] * G[2, t1, t_1] + K[i, 1, t1, t_1] * G[3, t1, t_1])
            Green[1, 1, i, t1, t_1] = (K[i, 2, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[1, t1, t_1])

    # plt.plot(t, np.real(Green[0, 0, 1, len(t)-1]), 'y--', t, np.imag(Green[0, 0, 1, len(t)-1]), 'k--')
    # plt.plot(t, np.real(Green[1, 1, 1, len(t)-1]), 'r--', t, np.imag(Green[1, 1, 1, len(t)-1]), 'b--')
    # plt.grid()
    # plt.show()

    print('1 - (Green_gtr + Green_les) for Spin Up site', i, 'is', 1 - np.real(Green[0, 0, i, len(t)-1, len(t)-1] + Green[1, 0, i, len(t)-1, len(t)-1]))
    print('1 - (Green_gtr + Green_les) for Spin Down site', i, 'is', 1 - np.real(Green[0, 1, i, len(t)-1, len(t)-1] + Green[1, 1, i, len(t)-1, len(t)-1]))

    print('                                                                                                                               ')

    print('Population for Spin Up les on site', i, 'is', Green[1, 0, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Down les on site', i, 'is', Green[1, 1, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Up gtr on site', i, 'is', Green[0, 0, i, len(t) - 1, len(t) - 1])
    print('Population for Spin Down gtr on site', i, 'is', Green[0, 1, i, len(t) - 1, len(t) - 1])

    print('                                                                                                                               ')

    # output
    # gtr_up = 'gtr_up_U={}_T={}_t={}_dt={}_A={}_lambda={}.out'
    # les_up = 'les_up_U={}_T={}_t={}_dt={}_A={}_lambda={}.out'
    # gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}_A={}_lambda={}.out'
    # les_down = 'les_down_U={}_T={}_t={}_dt={}_A={}_lambda={}.out'

    gtr_up = 'gtr_up_U={}_T={}_t={}_dt={}_A={}_lambda={}_ph.out'
    les_up = 'les_up_U={}_T={}_t={}_dt={}_A={}_lambda={}_ph.out'
    gtr_down = 'gtr_down_U={}_T={}_t={}_dt={}_A={}_lambda={}_ph.out'
    les_down = 'les_down_U={}_T={}_t={}_dt={}_A={}_lambda={}_ph.out'

    np.savetxt(gtr_up.format(U, T, tmax, dt, Amplitude, lambda_const), Green[0, 0, i].view(float), delimiter=' ')
    np.savetxt(les_up.format(U, T, tmax, dt, Amplitude, lambda_const), Green[1, 0, i].view(float), delimiter=' ')
    np.savetxt(gtr_down.format(U, T, tmax, dt, Amplitude, lambda_const), Green[0, 1, i].view(float), delimiter=' ')
    np.savetxt(les_down.format(U, T, tmax, dt, Amplitude, lambda_const), Green[1, 1, i].view(float), delimiter=' ')


########################################################################################################################
''' Main part starts here '''
# Amplitude_max = 0.5
# Amplitude_min = 0.0

U_max = 9.0
U_min = 8.0

######### perform loop over U #########
# for Amplitude in np.arange(Amplitude_min, Amplitude_max, 1.0):
for U in np.arange(U_min, U_max, 2.0):

    # U = 20.0
    U_ = np.zeros(len(t), float)
    Uc = 8.0

    Amplitude = 0.0

    for t_ in range(len(t)):
        U_[t_] = tune_U(t[t_])

    coupling = tune_Coupling(t)

    v_t = v_0 * np.exp(1j * Amplitude * np.cos(15 * t))     # time-dependent hopping

    # v_t = jv(0, 2)*np.exp(1j*0*np.cos(15*t))              # renormalized hopping

    # plt.plot(t, U_, 'r')
    # plt.plot(t, coupling, 'b')
    # plt.plot(t, v_t, 'k--')
    # plt.show()

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Starting DMFT loop for U =', U, '| Temperature =', T, '| time =', tmax, '| dt =', dt, '| A =', Amplitude, '| lambda = ', lambda_const)
    start = datetime.now()
    print('                                                                                                                               ')

    # Dot energy and Greens function
    E = np.zeros((4, len(t)), float)
    V_t = np.zeros((len(t), len(t)), complex)                   # electric field with Amplitude A generates a time-dependent hopping
    G_0 = np.zeros((4, len(t), len(t)), complex)                # Bare propagators for the 4 different states
    Green = np.zeros((2, 2, 4, len(t), len(t)), complex)        # Greens function with indices gtr/les | spin up/spin down | initial state | lower and upper branch time
    Green_old = np.zeros((2, 2, 4, len(t), len(t)), complex)

    # Hybridization to the bath
    Delta = np.zeros((2, 2, len(t), len(t)), complex)           # after the initial guess Delta becomes a two times function --> input into DeltaMatrix; indices are gtr/les | spin up/spin down | initial state
    DeltaMatrixG = np.zeros((4, 4, len(t), len(t)), complex)    # indices are initial and final states
    # DeltaMatrixK = np.zeros((4, 4, len(t), len(t)),complex)   # indices are initial and final states

    #  additional coupling of every lattice site to a fermionic/bosonic bath
    PhononBath = np.zeros((len(t), len(t)), complex)            # hybridization to a Phonon Bath
    FermionBath = np.zeros((len(t), len(t)), complex)           # hybridization to a Fermion Bath
    Coupling = np.zeros((len(t), len(t)), float)                # the coupling to the baths can be time-dependent

    initMatrix(Delta_init, phononBath, fermionBath, coupling, v_t)  # generate two-times matrices

    # Computation of bare propagators G_0 outside of Solver, so that this step is performed only once
    E[1] = -U_/2.0
    E[2] = -U_/2.0

    # # Computation of bare propagators G_0
    # for i in range(4):
    #     for t1 in range(len(t)):
    #         for t2 in range(t1+1):
    #             G_0[i, t1, t2] = np.exp(-1j * E[i, t1-t2] * t[t1-t2])

    for t1 in range(len(t)):
        for t2 in range(t1+1):
            G_0[0, t1, t2] = np.exp(-1j * 0)
            G_0[1, t1, t2] = np.exp(-1j * -integrate_U(t[t1], t[t2])/2)
            G_0[2, t1, t2] = np.exp(-1j * -integrate_U(t[t1], t[t2])/2)
            G_0[3, t1, t2] = np.exp(-1j * 0)

    # first DMFT loop with initial guess for Delta

    fillDeltaMatrixG(Delta)

    Solver(U, 1)

    counter = 0
    while np.amax(np.abs(Green_old - Green)) > 0.001:
        counter += 1

        # self-consistency condition for Bethe-lattice in initial Neel-state
        if fermion_bath == 1:
            Delta[:, 0] = V_t * Green[:, 1, 1] + Coupling * FermionBath
            Delta[:, 1] = V_t * Green[:, 0, 1] + Coupling * FermionBath

        else:
            Delta[:, 0] = V_t * Green[:, 1, 1]
            Delta[:, 1] = V_t * Green[:, 0, 1]

        Green_old[:] = Green

        fillDeltaMatrixG(Delta)

        Solver(U, 1)

        Diff = np.amax(np.abs(Green_old - Green))
        print('for U = ', U, ' and iteration Nr. ', counter, ' the Difference is ', Diff, ' after a calculation time ', datetime.now() - start)
        print('                                                                                                                               ')

    print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Computation of Greens functions for U =', U, '| Temperature =', T, '| time =', tmax, '| dt =', dt,
          '| lambda =', lambda_const, '| A =', Amplitude, 'finished after', counter,
          'iterations and', datetime.now() - start, 'seconds.')
    print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
