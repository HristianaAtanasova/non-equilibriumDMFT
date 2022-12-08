import numpy as np
import matplotlib.pyplot as plt

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t, lattice_structure):
    v = np.zeros((len(t), len(t)), complex)

    v_pump = np.exp(1j * 1 / ((1 + np.exp(10 * (t - t_pump_end))) * (1 + np.exp(-10 * (t - t_pump_start)))) * pumpA * np.cos(pumpOmega * t))
    v_probe = np.exp(1j * 1 / ((1 + np.exp(10 * (t - t_probe_end))) * (1 + np.exp(-10 * (t - t_probe_start)))) * probeA * np.cos(probeOmega * t))

    # v_t = v_0 * v_pump * v_probe    # time-dependent hopping
    v_t = v_0 * v_pump

    if lattice_structure == 1:
        for t1 in range(len(t)):
            for t2 in range(len(t)):
                v[t1, t2] = 1/2 * (v_t[t1]*np.conj(v_t[t2]) + v_t[t2]*np.conj(v_t[t1]))
        return v

    else:
        for t1 in range(len(t)):
            for t2 in range(len(t)):
                # v[t1, t2] = v_t[t1]*np.conj(v_t[t2])
                v[t1, t2] = tdiff(v_t, t2, t1)
        return v
