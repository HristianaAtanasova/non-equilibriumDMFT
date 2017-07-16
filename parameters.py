import numpy as np

## set parameters
# physical parameters
T = 0.1
beta = 1 / T
mu = 0
v_0 = 1
w_0 = 1

# numerical paramters
lambda_const = 0
delta_ = 0
rho = 6
wC = 10
v = 10
delta_width = 0.1
treshold = 1e-6

# time domain
tmax = 0.5
dt = 0.01
t = np.arange(0, tmax, dt)

# frequncy domain
dw = 0.01
wDOS = np.arange(-2 * v_0, 2 * v_0, dw)
Cut = np.pi / dt
w = np.arange(-Cut, Cut, dw)
fft_tmax = np.pi / dw
fft_tmin = -np.pi / dw
fft_dt = np.pi / Cut
fft_time = np.arange(fft_tmin, fft_tmax, fft_dt)