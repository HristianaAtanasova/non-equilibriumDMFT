from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime

# set parameters
T = 1
beta = 1 / T
mu = 0
wC = 10
t_param = 1
v = 10
treshold = 1e-6
tmax = 5
dt = 0.005
t = np.arange(0, tmax, dt)

########################################################################################################################
''' Calculate initial time-domain Hybridization function for given Density of States '''

dw = 0.01
wDOS = np.arange(-2 * t_param, 2 * t_param, dw)
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
    return 1 / (2 * np.pi * t_param ** 2) * np.sqrt(4 * t_param ** 2 - w ** 2)


# window function padded with zeros for semicircular DOS
N = int(2 * Cut / dw)
a = int(N / 2 + 2 * t_param / dw)
b = int(N / 2 - 2 * t_param / dw)
DOS = np.zeros(N + 1)
DOS[b:a] = semicircularDos(wDOS)

# frequency-domain Hybridization function
Hyb_les = DOS * fermi_function(w)
Hyb_gtr = DOS * (1 - fermi_function(w))

# Hyb_les = A(w) * fermi_function(w)
# Hyb_gtr = A(w) * (1 - fermi_function(w))

# obtain time-domain Hybridization function with fft
fDelta_les = np.conj((ifftshift(fft(fftshift(Hyb_les)))) * dw / np.pi)
# fDelta_les = (ifftshift(ifft(fftshift(Hyb_les)))) * 2/(fft_dt)
fDelta_gtr = (ifftshift(fft(fftshift(Hyb_gtr)))) * dw / np.pi

# get real times from fft_times
Delta_init = np.zeros((2, len(t)), complex)
for t_ in range(len(t)):
    # Delta_init[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
    # Delta_init[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
    Delta_init[0, t_] = fDelta_les[int(N / 2) + t_]
    Delta_init[1, t_] = fDelta_gtr[int(N / 2) + t_]

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


# Self energy for vertex functions
def SelfEnergy(K, DeltaMatrix, Sigma):
    for f in range(4):
        for j in range(4):
            Sigma[f] += K[j] * DeltaMatrix[f, j]


# fill in initial DeltaMatrix for two contour times with initial Delta for time-differences
def initDeltaMatrix(Delta):
    for t1 in range(len(t)):
        for t2 in range(len(t)):
            DeltaMatrix[0, 1, t1, t2] = tdiff(Delta[0], t1, t2)
            DeltaMatrix[0, 2, t1, t2] = tdiff(Delta[0], t1, t2)

            DeltaMatrix[1, 0, t1, t2] = tdiff(Delta[1], t1, t2)
            DeltaMatrix[1, 3, t1, t2] = tdiff(Delta[0], t1, t2)

            DeltaMatrix[2, 0, t1, t2] = tdiff(Delta[1], t1, t2)
            DeltaMatrix[2, 3, t1, t2] = tdiff(Delta[0], t1, t2)

            DeltaMatrix[3, 1, t1, t2] = tdiff(Delta[1], t1, t2)
            DeltaMatrix[3, 2, t1, t2] = tdiff(Delta[1], t1, t2)


def fillDeltaMatrix(Delta):
    # fill in DeltaMatrix for all times
    # initial dot state |0> can only go to |up> or |down> with a Delta_lesser
    DeltaMatrix[0, 1] = Delta[0]
    DeltaMatrix[0, 2] = Delta[0]

    # initial dot state |up> can only go to |0> or |up,down>
    DeltaMatrix[1, 0] = Delta[1]
    DeltaMatrix[1, 3] = Delta[0]

    # initial dot state |down> can only go to |0> or |up,down>
    DeltaMatrix[2, 0] = Delta[1]
    DeltaMatrix[2, 3] = Delta[0]

    # initial dot state |up,down> can only go to |up> or |down> with a Delta_greater
    DeltaMatrix[3, 1] = Delta[1]
    DeltaMatrix[3, 2] = Delta[1]


########## Impurity solver computes two-times correlation functions Green for a given hybridization Delta and interaction U ##########

def Solver(DeltaMatrix, U, init, Green_):

    ########## Computation of bold propagators on seperate branches of the contour ##########

    # set energy states
    epsilon = -U / 2.0
    E = [0, epsilon, epsilon, 2 * epsilon + U]

    # Initialize one branch propagators
    start = datetime.now()
    g_0 = np.zeros((4, len(t)), complex)  # As long as Impurity hamiltonian is time-independent G_0 depends only on time differences
    G_0 = np.zeros((4, len(t), len(t)), complex)  # g_0 as a two-times Matrix for convolution integrals
    G = np.zeros((4, len(t), len(t)), complex)  # indices are initial state, contour times located on the same branch t_n, t_m (propagation from t_m to t_n)
    Sigma = np.zeros((4, len(t), len(t)), complex)

    # Computation of bare propagators g_0
    for i in range(4):
        g_0[i] = (np.exp(-1j * E[i] * t))
        # extend g_0 to a two-times matrix G_0
        for t1 in range(len(t)):
            for t2 in range(t1+1):
                G_0[i, t1, t2] = g_0[i, t1-t2]


    # main loop over every pair of times t_n and t_m (located on the same contour-branch), where t_m is the smaller contour time

    # for t_n in range(len(t)):
    #     for t_m in range(t_n+1):
    #
    #         sum_t2 = np.zeros((4, t_n+1-t_m), complex)
    #         for t1 in range(t_m, t_n+1):
    #             sum_t2[:, t1-t_m] = np.trapz(Sigma[:, t1, t_m:t1+1]*G[:, t_m:t1+1, t_m])
    #
    #         sum_t1 = dt**2 * np.trapz(G_0[:, t_n, t_m:t_n+1]*sum_t2)
    #
    #         # Dyson equation for time (t_m, t_n)
    #         G[:, t_n, t_m] = G_0[:, t_n, t_m] - sum_t1
    #
    #         # Compute self-energy for time (t_m, t_n)
    #         Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrix[:, :, t_n, t_m], 1)


    # take integral over t1 outside the t_m loop
    for t_n in range(len(t)):
        sum_t1 = np.zeros((4, t_n+1), complex)
        for t_m in range(t_n, -1, -1):

            sum_t2 = dt**2 * np.trapz(G[:, t_m:t_n+1, t_m] * sum_t1[:, t_m:])

            # Dyson equation for time (t_m, t_n)
            G[:, t_n, t_m] = G_0[:, t_n, t_m] - sum_t2

            # Compute self-energy for time (t_m, t_n)
            Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrix[:, :, t_n, t_m], 1)

            sum_t1[:, t_m] = np.trapz(Sigma[:, t_m:t_n+1, t_m] * G_0[:, t_n, t_m:t_n+1])  # sum[:, t2=t_m]

    for t_n in range(len(t)):
        for t_m in range(t_n+1, len(t), +1):
                G[:, t_n, t_m] = np.conj(G[:, t_m, t_n])

    # for i in range(4):
    #     plt.plot(t, np.real(G[i, :, 0]), 'r--', t, np.imag(G[i, :, 0]), 'b--')
    #     plt.plot(t, np.real(G[i, 0, :]), 'r--', t, np.imag(G[i, 0, :]), 'b--')
    #     plt.show()

    print('Finished calculation of bold propagators after', datetime.now() - start)

    ########## Computation of Vertex Functions including hybridization lines between the upper and lower branch ##########

    start = datetime.now()
    K = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial state, contour times lower/upper branch
    K_0 = np.zeros((4, 4, len(t), len(t)), complex)

    # for every initial state i there is a set of 4 coupled equations for K
    for i in range(4):
        start_i = datetime.now()
        for f in range(4):
            K_0[i, f] = delta(i, f) * np.conj(G[i, :, None, 0]) * G[i, None, :, 0]

        # plt.plot(t, np.real(K_0[i, i, :, len(t) - 1]), 'r--', t, np.imag(K_0[i, i, :, len(t) - 1]), 'r--')
        # plt.plot(t, np.real(K_0[i, i, len(t) - 1]), 'b--', t, np.imag(K_0[i, i, len(t) - 1]), 'b--')
        # plt.show()

        Sigma = np.zeros((4, len(t), len(t)), complex)  # indices are final state, lower and upper branch time

        # for t_n in range(len(t)):
        #     for t_m in range(len(t)):
        #
        #         sum_t2 = np.zeros((4, t_n+1), complex)
        #         for t1 in range(t_n+1):
        #             sum_t2[:, t1] = np.trapz(Sigma[:, t1, :t_m+1] * G[:, t_m, :t_m+1])
        #
        #         sum_t1 = dt ** 2 * np.trapz(np.conj(G[:, t_n, :t_n+1]) * sum_t2)
        #
        #         K[i, :, t_n, t_m] = K_0[i, :, t_n, t_m] + sum_t1
        #
        #         SelfEnergy(K[i, :, t_n, t_m], DeltaMatrix[:, :, t_n, t_m], Sigma[:, t_n, t_m])

        for t_n in range(len(t)):
            sum_t1 = np.zeros((4, len(t)), complex)
            for t_m in range(len(t)):

                # print(t_n, t_m)
                sum_t2 = dt**2 * np.trapz(G[:, t_m, :t_m+1] * sum_t1[:, :t_m+1])

                # Dyson equation for time (t_m, t_n)
                K[i, :, t_n, t_m] = K_0[i, :, t_n, t_m] + sum_t2

                # Compute self-energy for time (t_m, t_n)
                SelfEnergy(K[i, :, t_n, t_m], DeltaMatrix[:, :, t_n, t_m], Sigma[:, t_n, t_m])

                sum_t1[:, t_m] = np.trapz(Sigma[:, :t_n+1, t_m] * np.conj(G[:, t_n, :t_n+1]))  # t_m = t_2

        # plt.plot(t, np.real(K[i, 1, :, len(t)-1]), 'r--', t, np.imag(K[i, 1, :, len(t)-1]), 'b--')
        # plt.plot(t, np.real(K[i, 1, len(t)-1]), 'y--', t, np.imag(K[i, 1, len(t)-1]), 'k--')
        # plt.show()

        print('Finished calculation of K for initial state', i, 'after', datetime.now() - start_i)
        err = np.abs(1 - np.abs(np.sum(K[0, :, len(t)-1, len(t)-1])))
        print('Error =', err)

        # output
        file = 'K_t={}_dt={}_i={}_f={}.out'
        for f in range(4):
            np.savetxt(file.format(tmax, dt, i, f), K[i, f].view(float), delimiter=' ')

    print('Finished calculation of all K after', datetime.now() - start)


    ########## Computation of two-times Green's functions ##########

    Green = np.zeros((2, 2, 4, len(t), len(t)), complex)  # Greens function with indices greater/lesser, spin up/spin down, initial state, lower and upper branch time
    for i in range(4):
        for t1 in range(len(t)):
            for t_1 in range(t1, -1, -1):

                Green[0, 0, i, t1, t_1] = K[i, 0, t1, t_1] * G[1, t1, t_1] + K[i, 2, t1, t_1] * G[3, t1, t_1]
                Green[1, 0, i, t1, t_1] = K[i, 1, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[2, t1, t_1]
                Green[0, 1, i, t1, t_1] = K[i, 0, t1, t_1] * G[2, t1, t_1] + K[i, 1, t1, t_1] * G[3, t1, t_1]
                Green[1, 1, i, t1, t_1] = K[i, 2, t1, t_1] * G[0, t1, t_1] + K[i, 3, t1, t_1] * G[1, t1, t_1]

                Green[0, 0, i, t_1, t1] = K[i, 0, t_1, t1] * G[1, t_1, t1] + K[i, 2, t_1, t1] * G[3, t_1, t1]
                Green[1, 0, i, t_1, t1] = K[i, 1, t_1, t1] * G[0, t_1, t1] + K[i, 3, t_1, t1] * G[2, t_1, t1]
                Green[0, 1, i, t_1, t1] = K[i, 0, t_1, t1] * G[2, t_1, t1] + K[i, 1, t_1, t1] * G[3, t_1, t1]
                Green[1, 1, i, t_1, t1] = K[i, 2, t_1, t1] * G[0, t_1, t1] + K[i, 3, t_1, t1] * G[1, t_1, t1]


        # output
        gtr = 'Green_gtr_t={}_dt={}_i={}.out'
        les = 'Green_les_t={}_dt={}_i={}.out'
        np.savetxt(gtr.format(tmax, dt, i), Green[0, 0, i].view(float), delimiter=' ')
        np.savetxt(les.format(tmax, dt, i), Green[1, 0, i].view(float), delimiter=' ')

    # summation over the initial state
    # Green_[0] = (Green[0, 0, 0] + Green[0, 0, 1] + Green[0, 0, 2] + Green[0, 0, 3]) / 4
    # Green_[1] = (Green[1, 0, 0] + Green[1, 0, 1] + Green[1, 0, 2] + Green[1, 0, 3]) / 4

    Green_[0] = Green[0, 0, init]  # spin up
    Green_[1] = Green[1, 0, init]  # spin up

    # for i in range(2):
    #     plt.plot(t, np.real(Green_[i, len(t)-1, ::-1]), 'r--', t, np.imag(Green_[i, len(t)-1, ::-1]), 'b--')
    #     plt.plot(t, np.real(Green_[i, ::-1, len(t)-1]), 'r--', t, np.imag(Green_[i, ::-1, len(t)-1]), 'b--')
    #     plt.show()

    return Green_


########################################################################################################################
''' Main part starts here '''
n_loops = 10
Umax = 6
Umin = 5
init = 0  # chose initial state

######### perform loop over U #########

for U in np.arange(Umin, Umax, 2):
    print('Starting DMFT loop for U =', U, '| Temperature =', T, '| time =', tmax, '| dt=', dt)
    start = datetime.now()

    Green_ = np.zeros((2, len(t), len(t)), complex)  # indices are greater/lesser, two contour times
    Green_old = np.zeros((2, len(t), len(t)), complex)
    DeltaMatrix = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial and final states, in general two times objects

    # first DMFT loop with initial guess for Delta
    initDeltaMatrix(Delta_init)
    Solver(DeltaMatrix, U, init, Green_)
    Delta = np.zeros((2, len(t), len(t)), complex)  # after the initial guess Delta becomes a two times function --> feed into Deltamatrix

    counter = 0
    while np.amax(np.abs(Green_old - Green_)) > 0.001:
        start = datetime.now()
        counter += 1
        Delta[0] = t_param ** 2 * Green_[1]
        Delta[1] = t_param ** 2 * Green_[0]
        fillDeltaMatrix(Delta)
        Green_old[:] = Green_
        Solver(DeltaMatrix, U, init, Green_)
        Diff = np.amax(np.abs(Green_old - Green_))
        print('for U = ', U, ' and iteration Nr. ', counter, ' the Difference is ', Diff, ' after a calculation time ', datetime.now() - start)


    print('Computation of Greens functions for U = ', U, 'finished after', counter, 'iterations and', datetime.now() - start, 'seconds.')


