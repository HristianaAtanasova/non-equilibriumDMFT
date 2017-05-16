import matplotlib.pyplot as plt
import numpy as np

tmax = 1
dt = 0.01
t = np.arange(0, tmax, dt)

Green_les = np.zeros((4, len(t), len(t)), complex)
Green_fft_les = np.zeros((4, len(t), len(t)), complex)

Green_gtr = np.zeros((4, len(t), len(t)), complex)
Green_fft_gtr = np.zeros((4, len(t), len(t)), complex)

SpinUpPop = np.zeros((len(t)), complex)

les = 'Green_les_t={}_dt={}_i={}.out'
fft_les = 'Green_fft_les_t={}_dt={}_i={}.out'

gtr = 'Green_gtr_t={}_dt={}_i={}.out'
fft_gtr = 'Green_fft_gtr_t={}_dt={}_i={}.out'

for i in range(4):
    Green_les[i] = np.loadtxt(les.format(tmax,dt,i)).view(complex)
    Green_fft_les[i] = np.loadtxt(fft_les.format(tmax,dt,i)).view(complex)

    Green_gtr[i] = np.loadtxt(gtr.format(tmax,dt,i)).view(complex)
    Green_fft_gtr[i] = np.loadtxt(fft_gtr.format(tmax,dt,i)).view(complex)

i = 0
SpinUpPop = Green_les[i].diagonal()
# plt.plot(t, np.real(SpinUpPop), 'b--', t, np.imag(SpinUpPop), 'r--', label='$dt = {dt}$'.format(dt=dt))

# plt.plot(t, np.real(Green_les[0, len(t)-1, ::-1]), 'b--', t, np.imag(Green_les[0, len(t)-1, ::-1]), 'b--', label='timedep')
# plt.plot(t, np.real(Green_fft_les[0, :, len(t)-1]), 'r--', t,  np.imag(Green_fft_les[0, :, len(t)-1]), 'r--', label='fft')

plt.plot(t, np.real(Green_gtr[0, len(t)-1, ::-1]), 'b--', t, np.imag(Green_gtr[0, len(t)-1, ::-1]), 'b--', label='timedep')
plt.plot(t, np.real(Green_fft_gtr[0, :, len(t)-1]), 'r--', t,  np.imag(Green_fft_gtr[0, :, len(t)-1]), 'r--', label='fft')

plt.grid()
plt.legend()
plt.show()
