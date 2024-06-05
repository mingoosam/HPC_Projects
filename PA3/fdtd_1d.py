#### 1D FDTD in free space

import time
from cupyx.profiler import benchmark
import os
import csv
import numpy as np
from math import exp
from math import ceil 
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cupy as cp
import numpy as np

from support import write_stats, plot_output_image, load_config, update_params

def fdtd(params):

    device = params['device']
    nsteps = params['nsteps']
    ke = params['ke']
    kc = params['kc']
    spread = params['spread']
    t0 = params['t0']

    if device == 'cpu':

        ex = params['ex']
        hy = params['hy']
        for time_step in range(1, nsteps+1):

            # Calculate Ex
            for k in range(1, ke):
                ex[k] = ex[k] + 0.5 * (hy[k - 1] - hy[k])

            # Introduce Gaussian pulse in center
            pulse = exp(-0.5 * ((t0 - time_step) / spread) ** 2)
            ex[kc] = pulse

            # Calculate Hy
            for k in range(ke - 1):
                hy[k] = hy[k] + 0.5 * (ex[k] - ex[k + 1])

    elif device == 'gpu':
       
        ex = cp.asarray(params['ex'])
        hy = cp.asarray(params['hy'])
        
        for time_step in range(1, nsteps+1):

            # Calculate Ex
            ex[1:ke] += 0.5 * (hy[:ke - 1] - hy[1:ke])

            # Introduce Gaussian pulse in center
            pulse = cp.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
            ex[kc] = pulse

            # Calculate Hy
            hy[:ke-1] += 0.5 * (ex[:ke-1] - ex[1:ke])
            
        ex = ex.get()
        hy = hy.get()
        
    return ex, hy, pulse

def run_fdtd(params):

    start_time = time.time()
    
    ex, hy, pulse = fdtd(params)

    execution_time = time.time() - start_time
    print(f"execution time = {execution_time} seconds")

    write_stats(params, execution_time)
    
    plot_output_image(params, ex, hy)


if __name__=="__main__":

    params = load_config(sys.argv)
    params = update_params(params)

    start_time = time.time()
    
    run_fdtd(params)
 
    execution_time = time.time() - start_time
    print(f"execution time = {execution_time} seconds")
    
