import numpy as np

def genv(pumpA, pumpOmega, probeA, probeOmega, v_0, t):
    v = np.zeros((len(t), len(t)), complex)
    v_t = v_0 * np.exp(1j * pumpA * np.cos(pumpOmega * t)) * np.exp(1j * probeA * np.cos(probeOmega * t))    # time-dependent hopping
    for t1 in range(len(t)):
        for t2 in range(len(t)):
            v[t1, t2] = v_t[t1]*np.conj(v_t[t2])
    return v
# np.savez_compressed('timedepHopping', t=t, v=v)
