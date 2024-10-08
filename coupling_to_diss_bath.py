import numpy as np
import matplotlib.pyplot as plt


def gen_timedepLambda(t, t_diss_end, Lambda):
    Lambda_t = np.zeros((len(t), len(t)), float)
    lambda_t = np.sqrt(Lambda) / (1 + np.exp(10 * (t - t_diss_end)))

    for t1 in range(len(t)):
        for t2 in range(len(t)):
            Lambda_t[t1, t2] = lambda_t[t1]*lambda_t[t2]

    return Lambda_t

# t = np.arange(0, 10, 0.01)
# Lambda = 0.5
# t_diss_end = 15
#
# L = gen_timedepLambda(t, t_diss_end, Lambda)
# plt.plot(t, L[:,-1], 'b')
# plt.show()
