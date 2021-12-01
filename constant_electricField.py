import numpy as np
import matplotlib.pyplot as plt


def genv(A_pump, v_0, t, lattice_structure):
    v = np.zeros((len(t), len(t)), complex)
    # r = (24.0 * np.pi * (t / 3.0) - 27.0 * np.sin(np.pi * t / 3.0) + np.sin(np.pi * 3.0 * t / 3.0)) / (48.0*np.pi)
    r = 1.0 - 1.0 / (1.0 + np.exp((t - 3.0)*4.0)) 
    v_t = v_0 * np.exp(1j * A_pump *  r)
    # v_t = v_0 * np.exp(1j * A_pump * t)
    
    plt.plot(t, r, label='r')
    plt.plot(t, np.real(v_t), '-', t, np.imag(v_t), '--', label='v_t')
    plt.legend()
    plt.savefig('electric_field.pdf')
    plt.close()

    if lattice_structure == 1:
        for t1 in range(len(t)):
            for t2 in range(len(t)):
                v[t1, t2] = 1/2 * (v_t[t1]*np.conj(v_t[t2]) + v_t[t2]*np.conj(v_t[t1]))
        return v

    else:
        for t1 in range(len(t)):
            for t2 in range(len(t)):
                v[t1, t2] = v_t[t1]*np.conj(v_t[t2])
        return v

