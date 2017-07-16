from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime
from scipy.integrate import quad
from scipy.signal import hilbert
from numpy.linalg import inv
from numpy.linalg import solve

########################################################################################################################
''' Time-dependent coupling to the phonon Bath lambda | hopping v | interaction U '''

t_turn = 2
t0 = 0.1
F_0 = 0

def tune_Coupling(t):
    return lambda_const / (1 + np.exp(10 * (t - 10)))


def tune_U(x):
    if t_turn <= x <= t0+t_turn:
        return U + (Uc-U)*(1/2 - 3/4 * np.cos(np.pi*(x-t_turn)/t0) + 1/4 * (np.cos(np.pi*(x-t_turn)/t0))**3)
    elif x < t_turn:
        return U
    return Uc


def tune_Hopping(x):
    if t_turn_on <= x <= t0+t_turn:
        return 1/2 - 3/4 * np.cos(np.pi*(x-t_turn)/t0) + 1/4 * (np.cos(np.pi*(x-t_turn)/t0))**3
    elif x < t_turn:
        return 0
    else:
        return 1


def integrate_F(t):
    return F_0*quad(rampHopping,0,t)[0]


########################################################################################################################
''' Main part starts here '''

Umax = 6
Umin = 5

######### perform loop over U #########
for U in np.arange(Umin, Umax, 2):

    U_ = np.zeros(len(t), float)
    Uc = 5

    for t_ in range(len(t)):
        U_[t_] = tune_U(t[t_])

    phononCoupling = tune_Coupling(t)

    plt.plot(t, U_, 'r--')
    plt.plot(t, phononCoupling, 'b--')
    plt.show()

    print('-------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Starting DMFT loop for U =', U, '| Temperature =', T, '| time =', tmax, '| dt =', dt, '| turn on =', t_turn, '| lambda = ', lambda_const)
    start = datetime.now()
    print('                                                                                                                               ')

    Green = np.zeros((2, 2, 4, len(t), len(t)), complex)  # Greens function with indices gtr/les | spin up/spin down | initial state | lower and upper branch time
    Green_old = np.zeros((2, 2, 4, len(t), len(t)), complex)

    Delta = np.zeros((2, 2, 2, len(t), len(t)), complex)  # after the initial guess Delta becomes a two times function --> input into DeltaMatrix; indices are gtr/les | spin up/spin down | initial state

    PhononBath = np.zeros((len(t), len(t)), complex)
    PhononCoupling = np.zeros((len(t), len(t)), float)

    # Delta, PhononBath, PhononCoupling = Hybridization.generate()

    # first DMFT loop with initial guess for Delta
    Solver(U, 1)

    counter = 0
    while np.amax(np.abs(Green_old - Green)) > 0.001:
        counter += 1

        # self-consistency condition for Bethe-lattice in initial Neel-state
        Delta[:, 0, 1] = v_0 * Green[:, 1, 1] * v_0
        Delta[:, 1, 1] = v_0 * Green[:, 0, 1] * v_0

        Delta[:, 0, 0] = v_0 * Green[:, 1, 1] * v_0 + PhononCoupling[:, :] * PhononBath
        Delta[:, 1, 0] = v_0 * Green[:, 0, 1] * v_0 + PhononCoupling[:, :] * PhononBath


        Green_old[:] = Green

        # gf = impurity_solver(U, state, Hybridization, params)

        Solver(U, 1, Delta)

        Diff = np.amax(np.abs(Green_old - Green))
        print('for U = ', U, ' and iteration Nr. ', counter, ' the Difference is ', Diff, ' after a calculation time ', datetime.now() - start)
        print('                                                                                                                               ')


    print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Computation of Greens functions for U =', U, '| Temperature =', T, '| time =', tmax, '| dt =', dt,
          '| lambda =', lambda_const, '| turn on =', t_turn, 'finished after', counter,
          'iterations and', datetime.now() - start, 'seconds.')
    print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
