import numpy as np

def genv(pumpA, pumpOmega, t_pump_start, t_pump_end, probeA, probeOmega, t_probe_start, t_probe_end, v_0, t):
    v = np.zeros((len(t), len(t)), complex)

    v_pump = 1 / ((1 + np.exp(10 * (t - t_pump_end))) * (1 + np.exp(-10 * (t + t_pump_start)))) * np.exp(1j * pumpA * np.cos(pumpOmega * t))
    v_probe = 1 / ((1 + np.exp(10 * (t - t_probe_end))) * (1 + np.exp(-10 * (t + t_probe_start)))) * np.exp(1j * probeA * np.cos(probeOmega * t))

    v_t = v_0 * v_pump * v_probe    # time-dependent hopping

    for t1 in range(len(t)):
        for t2 in range(len(t)):
            v[t1, t2] = v_t[t1]*np.conj(v_t[t2])
    return v

# def genv(pumpA, pumpOmega, probeA, probeOmega, v_0, t):
#     v = v_0 * np.exp(1j * pumpA * np.cos(pumpOmega * t)) * np.exp(1j * probeA * np.cos(probeOmega * t))
#     return v

# np.savez_compressed('timedepHopping', t=t, v=v)
