import numpy as np
import matplotlib.pyplot as plt

def genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t, lattice_structure):
    v = np.zeros((len(t), len(t)), complex)

    v_pump = np.exp(1j * 1 / ((1 + np.exp(10 * (t - t_pump_end))) * (1 + np.exp(-10 * (t - t_pump_start)))) * pumpA * np.cos(pumpOmega * t))
    v_probe = np.exp(1j * 1 / ((1 + np.exp(10 * (t - t_probe_end))) * (1 + np.exp(-10 * (t - t_probe_start)))) * probeA * np.cos(probeOmega * t))

    v_t = v_0 * v_pump * v_probe    # time-dependent hopping

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

# t = np.arange(0, 10, 0.01)
# pumpA = 1.0
# pumpOmega = 5.0
# t_pump_start = 0.0
# t_pump_end = 5.0
# probeA = 0.05
# probeOmega = 2.5
# t_probe_start = 5.0
# t_probe_end = 15.0
# v_0 = 1
#
# v = genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t)
# plt.plot(t, v, 'b')
# plt.show()
