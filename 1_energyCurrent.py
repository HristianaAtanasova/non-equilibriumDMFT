#!/usr/bin/env python
from figures import *

import argparse
import toml
import zipfile
import numpy as np
import os

Path = os.getcwd() + '/U5.0_probeA{}_probeOmega{}_pumpA{}_pumpOmega{}/T{}'

pumpOmega_ = [2.5]
probeOmega_ = [2.5]
probeA_ = [0.05]
pumpA_ = [2.0]
Temperature = [1.0]

for pumpOmega in pumpOmega_:
    for probeOmega in probeOmega_:
        for T in Temperature:
            for probeA in probeA_:
                # Create a figure with sane defaults for publication.
                fig = create_figure()
                ax = create_single_panel(fig, xlabel=r'$\mathrm{t} \mathrm{\gamma}$', ylabel=r'$\mathrm{I_\mathrm{E}(\mathrm{t} \mathrm{\gamma})}$',
                palette=('viridis'), numcolors=(7))
                for pumpA in pumpA_:
                    msg = 'Executing pumpOmega={} | pumpA = {} | probeOmega={} | probeA = {} | T = {}'
                    print(msg.format(pumpOmega, pumpA, probeOmega, probeA, T))

                    path = Path.format(probeA, probeOmega, pumpA, pumpOmega, T)
                    parser = argparse.ArgumentParser(description = "run dmft")
                    parser.add_argument("--params",   default = path + "/run.toml")
                    args = parser.parse_args()

                    with open(args.params, "r") as f:
                        params = toml.load(f)

                    params.update(vars(args))

                    loaded = np.load(path + '/energyCurrent.npz')
                    t = loaded['t']
                    current = loaded['energyCurrent']
                    dt = t[1]-t[0]

                    ax.plot(t, np.real(current[1]+current[0]), label='$A/ \gamma = {}$'.format(pumpA))

                title = 'energyCurrent_probeA={}_probeOmega={}_pumpOmega={}_T={}.pdf'
                # Clean up and save:
                finalize_and_save(fig, title.format(params['probeA'],params['probeOmega'], params['pumpOmega'], params['T']))
