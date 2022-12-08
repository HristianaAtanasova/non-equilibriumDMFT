import numpy as np
import matplotlib.pyplot as plt

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])


def genv(A_pump, v_0, t, t_pump_start, t_pump_end, lattice_structure):
    v = np.zeros((len(t), len(t)), complex)
    # r = (24.0 * np.pi * (t / 3.0) - 27.0 * np.sin(np.pi * t / 3.0) + np.sin(np.pi * 3.0 * t / 3.0)) / (48.0*np.pi)
    # r = 1.0 - 1.0 / (1.0 + np.exp((t - 3.0)*10.0)) 
    # v_t = v_0 * np.exp(1j * 1.0 / ((1.0 + np.exp(0.5 * (t - t_pump_end))) * (1.0 + np.exp(-0.5 * (t - t_pump_start)))) * A_pump * t)
    # v_t = v_0 * np.exp(-1j * 1/2 * (A_pump)**2  *  t)
    v_t = v_0 * np.exp(-1j * A_pump  *  t)

    if lattice_structure == 1:
        for t1 in range(len(t)):
            for t2 in range(len(t)):
                v[t1, t2] = 1/2 * (v_t[t1]*np.conj(v_t[t2]) + v_t[t2]*np.conj(v_t[t1]))
                # v[t2, t1] = 1/2 * tdiff(v_t, t2, t1) + 1/2 * tdiff(v_t, t1, t2)
        return v

    else:
        for t1 in range(len(t)):
            for t2 in range(len(t)):
                v[t1, t2] = v_t[t1]*np.conj(v_t[t2])
                # v[t2, t1] = tdiff(v_t, t2, t1)
        return v

