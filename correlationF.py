import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 6, 0.001)
G0 = np.loadtxt("Green_100_U=6_T=1_time6.out").view(complex)
G1 = np.loadtxt("Green_101_U=6_T=1_time6.out").view(complex)
G2 = np.loadtxt("Green_102_U=6_T=1_time6.out").view(complex)
G3 = np.loadtxt("Green_103_U=6_T=1_time6.out").view(complex)

plt.plot(t, np.real(G0[:, len(t)-1]), 'r--')
plt.plot(t, np.imag(G0[:, len(t)-1]), 'r--', label='init state 0')
plt.plot(t, np.real(G1[:, len(t)-1]), 'b--')
plt.plot(t, np.imag(G1[:, len(t)-1]), 'b--', label='init state 1')
plt.plot(t, np.real(G2[:, len(t)-1]), 'g--')
plt.plot(t, np.imag(G2[:, len(t)-1]), 'g--', label='init state 2')
plt.plot(t, np.real(G3[:, len(t)-1]), 'k--')
plt.plot(t, np.imag(G3[:, len(t)-1]), 'k--', label='init state 3')

# plt.plot(t, np.real(G0[:, int(len(t)/2)]), 'r--', t, np.imag(G0[:, int(len(t)/2)]), 'r--', label='init state 0')
# plt.plot(t, np.real(G1[:, int(len(t)/2)]), 'b--', t, np.imag(G1[:, int(len(t)/2)]), 'b--', label='init state 1')
# plt.plot(t, np.real(G2[:, int(len(t)/2)]), 'g--', t, np.imag(G2[:, int(len(t)/2)]), 'g--', label='init state 2')
# plt.plot(t, np.real(G3[:, int(len(t)/2)]), 'k--', t, np.imag(G3[:, int(len(t)/2)]), 'k--', label='init state 3')

plt.grid()
plt.legend()
plt.show()
