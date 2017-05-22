import matplotlib.pyplot as plt
import numpy as np

tmax = 2
dt = 0.01
t = np.arange(0, tmax, dt)

Kfft = np.zeros((4, 4, len(t), len(t)), complex)
K = np.zeros((4, 4, len(t), len(t)), complex)
Up_fft = np.zeros((len(t)), complex)
Up_timedep = np.zeros((len(t)), complex)

file_fft = 'Kfft_t={}_dt={}_i={}_f={}.out'
file_timedep = 'K_t={}_dt={}_i={}_f={}.out'

for i in range(4):
    for f in range(4):
        Kfft[i, f] = np.loadtxt(file_fft.format(tmax,dt,i,f)).view(complex)

for i in range(4):
    for f in range(4):
        K[i, f] = np.loadtxt(file_timedep.format(tmax,dt,i,f)).view(complex)
i = 0
Up_timedep = K[i,1].diagonal() + K[i,3].diagonal()
Up_fft = Kfft[i,1].diagonal() + Kfft[i,3].diagonal()

# Up_fft3 = K3[0,1].diagonal() + K3[0,3].diagonal()
# Up_timedep3 = Kfft3[0,1].diagonal() + Kfft3[0,3].diagonal()

plt.plot(t, np.real(Up_fft), 'b--', label='$dt = {dt}$'.format(dt=dt))
plt.plot(t, np.real(Up_timedep), 'r--', label='$dt = {dt}$'.format(dt=dt))

plt.grid()
plt.legend()
plt.show()
