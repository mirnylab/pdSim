#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pmap import pmap
import os

resolution = 2 # 15

try:
    job_ID = os.environ["SLURM_ARRAY_TASK_ID"]
    print("This is task # {:} in a SLURM Job Array".format(job_ID))
except KeyError:
    print("This is a single job.")
    job_ID = 1

sd_array = np.logspace(-2, -0.5, resolution)
sp_array = np.logspace(-4, -2, resolution)

params = [(sd, sp) for sd in sd_array for sp in sp_array]

def simulation_wrapper(sd_sp_tuple):
    from simple import simplest_simulations
    return simplest_simulations(*sd_sp_tuple, trials=1, map=map)

output = pmap(simulation_wrapper, params)

df = pd.concat(dict(zip(params, output)), names=['sd', 'sp'])
df.to_csv('SimSweep{:}.csv.gz'.format(job_ID), compression='gzip')

