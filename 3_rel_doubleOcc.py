#!/usr/bin/env python
from figures import *

from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

#!/usr/bin/env python
from figures import *
import argparse
import toml
import zipfile
import numpy as np
import os

Path = os.getcwd() + '/U5.0_probeA{}_probeOmega{}_pumpA{}_pumpOmega{}/T{}'

pumpOmega_ = [3.0]
# probeOmega_ = [1.25, 1.5, 2.5, 3.0, 3.75, 5.0, 7.5]
# probeOmega_ = [2.5, 3.75, 5.0, 7.5]
probeOmega_ = [2.5, 3.0, 5.0, 7.5]
probeA_ = [0.05]
pumpA_ = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
Temperature = [0.06, 1.0]

for pumpOmega in pumpOmega_:
    for probeA in probeA_:
        for T in Temperature:
            # Create a figure with sane defaults for publication.
            fig = create_figure()
            axes = create_quad_split(fig,
                                   xlabel=(r'$\mathrm{t} $', r'$\mathrm{t} $'),
                                   ylabel=(r'$\mathrm{\mathrm{\partial_\mathrm{A_\mathrm{probe}}}d(t)}$', r'$\mathrm{\mathrm{\partial_\mathrm{A_\mathrm{probe}}}d(t)}$'),
                                   palette=('viridis','viridis','viridis','viridis'), numcolors=(7, 7, 7, 7)
                                   )
            for i, ax in enumerate(axes):
                probeOmega = probeOmega_[i]
                for pumpA in pumpA_:
                    msg = 'Executing pumpOmega={} | pumpA = {} | probeOmega={} | probeA = {} | T = {}'
                    print(msg.format(pumpOmega, pumpA, probeOmega, probeA, T))

                    path = Path.format(probeA, probeOmega, pumpA, pumpOmega, T)
                    path_to_probeA0 = Path.format(0.0, 1.25, pumpA, pumpOmega, T)
                    parser = argparse.ArgumentParser(description = "run dmft")
                    parser.add_argument("--params",   default = path + "/run.toml")
                    args = parser.parse_args()

                    with open(args.params, "r") as f:
                        params = toml.load(f)

                    params.update(vars(args))

                    Vertex = path_to_probeA0 + '/K_1_f_T={}.npz'
                    loaded = np.load(Vertex.format(T))
                    K_probe0 = loaded['K']

                    Vertex = path + '/K_1_f_T={}.npz'
                    loaded = np.load(Vertex.format(T))
                    t = loaded['t']
                    K = loaded['K']
                    dt = t[1]-t[0]

                    if probeOmega == probeOmega_[3]:
                        ax.plot(t, np.real((K[3].diagonal() - K_probe0[3].diagonal())/probeA), label='$A={}$'.format(pumpA))
                    else:
                        ax.plot(t, np.real((K[3].diagonal() - K_probe0[3].diagonal())/probeA))

                title = '$\mathrm{\omega_\mathrm{probe}}$' + '$={}$'.format(probeOmega)
                ax.text(0.8, 0.9, title, fontsize='x-small', verticalalignment='center', horizontalalignment='center', transform=ax.transAxes)

            # Clean up and save:
            title = 'quadrel_doubleOcc_probeA={}_last_probeOmega={}_pumpOmega={}_T={}.pdf'
            finalize_and_save(fig, title.format(probeA, probeOmega, pumpOmega, T))
