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
tmax = 2
dt = 0.01
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
Delta = np.zeros((2, len(t)), complex)
for t_ in range(len(t)):
    # Delta[0, t_] = fDelta_les[int((N-len(t))/2) + t_]
    # Delta[1, t_] = fDelta_gtr[int((N-len(t))/2) + t_]
    Delta[0, t_] = fDelta_les[int(N / 2) + t_]
    Delta[1, t_] = fDelta_gtr[int(N / 2) + t_]

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


def tdiff(D, t1, t2):
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])


def delta(f, i):
    return 1 if f == i else 0


# Self energy for vertex functions
def Vertex(K, DeltaMatrix, i, V):
    for f in range(4):
        for j in range(4):
            V[f] += K[i, j] * DeltaMatrix[f, j]
    return V


# Integral equation for vertex functions
def Dyson(V, G, i, K):
    for f in range(4):
        Conv = np.zeros((len(t), len(t)), complex)
        for t2 in range(len(t)):
            Conv[:, t2] = trapezConv(V[f, :, t2], G[f, :])
        for t1 in range(len(t)):
            K[i, f, t1, :] += trapezConv(np.conj(G[f, :]), Conv[t1])
    return K[i]

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
    G_0 = np.zeros((4, len(t)), complex)  # As long as Impurity hamiltonian is time-independent G_0 depends only on time differences
    G = np.zeros((4, len(t), len(t)), complex)  # indices are initial state, contour times located on the same branch t_n, t_m (propagation from t_m to t_n)
    Sigma = np.zeros((4, len(t), len(t)), complex)

    # Computation of bare propagators G_0 and initialization of G to 1 for diagonal times
    for i in range(4):
        G_0[i] = (np.exp(-1j * E[i] * t))
        np.fill_diagonal(G[i], 1)

    # Initialize Sigma for diagonal times
    Sigma = np.sum(G[None] * DeltaMatrix, 1)

    # main loop over every pair of times t_n and t_m (located on the same contour-branch), where t_m is the smaller contour time
    for t_n in range(len(t)):
        # propagate from t_n to t_n+1
        sum_t2 = np.zeros((4, len(t), len(t)), complex)
        for t1 in range(t_n):
            for t2 in range(t1):
                sum_t2[:, t1, 0] += Sigma[:, t1, t2] * G[:, t2, 0]

        sum_t1 = np.zeros((4, len(t), len(t)), complex)
        for t1 in range(t_n):
            sum_t1[:, t_n, 0] += dt ** 2 * G_0[:, t_n - t1] * sum_t2[:, t1, 0]
        
        # Dyson equation for (t_n, t_0)
        G[:, t_n, 0] = G_0[:, t_n] - sum_t1[:, t_n, 0]
        
        # Self-energy for (t_n, t_0)
        Sigma[:, t_n, 0] = np.sum(G[None, :, t_n, 0] * DeltaMatrix[:, :, t_n, 0], 1)

        # propagate time slice for t_n
        for t_m in range(t_n):
            sum_t2 = np.zeros((4, len(t), len(t)), complex)
            for t1 in range(t_m, t_n):
                for t2 in range(t_m, t1):
                    sum_t2[:, t1, t_m] += Sigma[:, t1, t2] * G[:, t2, t_m]

            sum_t1 = np.zeros((4, len(t), len(t)), complex)
            for t1 in range(t_m, t_n):
                sum_t1[:, t_n, t_m] += dt ** 2 * G_0[:, t_n - t1] * sum_t2[:, t1, t_m]

            G[:, t_n, t_m] = G_0[:, t_n - t_m] - sum_t1[:, t_n, t_m]

            # Sum = np.zeros((4, len(t), len(t)), complex)
            # for t1 in range(t_m, t_n):
            #     for t2 in range(t_m, t1):
            #         Sum[:, t_n, t_m] += dt**2 * G_0[:, t_n - t1, None] * Sigma[:, t1, t2] * G[:, t2, t_m]
            #
            # G[:, t_n, t_m] = G_0[:, t_n - t_m, None] - Sum[:, t_n, t_m]

            # Compute self-energy for time t_m, t_n
            Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrix[:, :, t_n, t_m], 1)

    plt.plot(t, np.real(G[0, len(t)-1, ::-1]), 'r--', t, np.imag(G[0, len(t)-1, ::-1]), 'b--')
    plt.show()
    ########## Computation of Vertex Functions including hybridization lines between the upper and lower branch ##########

    K = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial, contour times on upper and lower branch
    K_0 = np.zeros((4, 4, len(t), len(t)), complex)

    # Computation of K_0
    for i in range(4):
        for f in range(4):
            K_0[i, f] = delta(i, f) * np.conj(G[i]) * G[i]



    ########## Computation of two-times Green's functions ##########

    Green = np.zeros((2, 2, 4, len(t), len(t)), complex)  # Greens function with indices greater/lesser, spin up/spin down, initial state, upper and lower branch time
    for i in range(4):
        for t1 in range(len(t)):
            for t_1 in range(t1 + 1):
                Green[0, 0, i, t_1, t1] = K[i, 0, (t1 - t_1), t1] * G[1, t_1] + K[i, 2, (t1 - t_1), t1] * G[3, t_1]
                Green[1, 0, i, t_1, t1] = K[i, 1, (t1 - t_1), t1] * G[0, t_1] + K[i, 3, (t1 - t_1), t1] * G[2, t_1]
                Green[0, 1, i, t_1, t1] = K[i, 0, (t1 - t_1), t1] * G[2, t_1] + K[i, 1, (t1 - t_1), t1] * G[3, t_1]
                Green[1, 1, i, t_1, t1] = K[i, 2, (t1 - t_1), t1] * G[0, t_1] + K[i, 3, (t1 - t_1), t1] * G[1, t_1]

    Green_[0] = (Green[0, 0, 0] + Green[0, 0, 1] + Green[0, 0, 2] + Green[0, 0, 3]) / 4  # the summation is performed over the spin up index, we have spin-symmetry
    Green_[1] = (Green[1, 0, 0] + Green[1, 0, 1] + Green[1, 0, 2] + Green[1, 0, 3]) / 4

    # Green_[0] = Green[0, 0, init]
    # Green_[1] = Green[1, 0, init]

    return Green_


########################################################################################################################
''' Main part starts here '''
n_loops = 10
Umax = 5.0
Umin = 4.0
init = 0  # chose initial state

######### perform loop over U #########

for U in np.arange(Umin, Umax, 1.00):
    print('Starting DMFT loop for U =', U)
    start = datetime.now()
    Green_ = np.zeros((2, len(t), len(t)), complex)  # indices are greater/lesser, two contour times
    Green_old = np.zeros((2, len(t), len(t)), complex)
    DeltaMatrix = np.zeros((4, 4, len(t), len(t)), complex)  # indices are initial and final states, in general two times objects

    # fill in initial DeltaMatrix for two contour times with initial Delta for time-differences
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

    # first DMFT loop with initial guess for Delta
    Solver(DeltaMatrix, U, init, Green_)
    Delta = np.zeros((2, len(t), len(t)), complex)  # after the initial guess Delta becomes a two times function

    # # output
    # np.savetxt("Green_les_U="+str(U)+"_T="+str(T)+"_t="+str(tmax)+".out", Green_[1].view(float), delimiter=' ')
    # np.savetxt("Green_gtr_U="+str(U)+"_T="+str(T)+"_t="+str(tmax)+".out", Green_[0].view(float), delimiter=' ')

    # plt.plot(t, np.real(Green_[0]), 'r--', label='Green_gtr_r')
    # plt.plot(t, np.imag(Green_[0]), 'b--', label='Green_gtr_i')
    # plt.plot(t, np.real(Green_[1]), 'y--', label='Green_les_r')
    # plt.plot(t, np.imag(Green_[1]), 'k--', label='Green_les_i')
    # plt.legend()
    # plt.grid()
    # plt.show()

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

        # output
        np.savetxt("Green_les_U=" + str(U) + "_T=" + str(T) + "_t=" + str(tmax) + ".out", Green_[1].view(float), delimiter=' ')
        np.savetxt("Green_gtr_U=" + str(U) + "_T=" + str(T) + "_t=" + str(tmax) + ".out", Green_[0].view(float), delimiter=' ')

        # plt.plot(t, np.real(Green_[0]), 'r--', label='Green_gtr_r')
        # plt.plot(t, np.imag(Green_[0]), 'b--', label='Green_gtr_i')
        # plt.plot(t, np.real(Green_[1]), 'y--', label='Green_les_r')
        # plt.plot(t, np.imag(Green_[1]), 'k--', label='Green_les_i')
        # plt.legend()
        # plt.grid()
        # plt.show()

    print('Computation of Greens functions for U = ', U, 'finished after', counter, 'iterations and', datetime.now() - start, 'seconds.')


