import scipy
import sys

import fdtd_1d 
import fdtd_animate
from support import load_config, update_params
from timing import plot_timing_stats

def run(params):

    if params['animate'] == 0:
        fdtd_1d.run_fdtd(params)

    elif params['animate'] == 1:
        fdtd_animate.run_animation(params)

if __name__=="__main__":

    params = load_config(sys.argv)

    #grid_sizes = [100, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000]
    #grid_sizes = [1000000, 10000000, 100000000, 1000000000]

    grid_sizes = [100, 300, 500]
    for grid_size in grid_sizes:
        
        params['ke'] = grid_size
        params = update_params(params)
        run(params)

    plot_timing_stats(params) 
