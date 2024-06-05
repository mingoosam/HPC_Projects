import time
import os
import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

## Animation methods ##

def init(params):
    
    ex = params['ex']
    hy = params['hy']

    line_ex.set_ydata([np.nan] * len(ex))
    line_hy.set_ydata([np.nan] * len(hy))

    return line_ex, line_hy

def animate(frame,params):

    courant_number = params['courant_number']
    ex = params['ex']
    hy = params['hy']
    kc = params['kc']
    ke = params['ke']
    t0 = params['t0']
    spread = params['spread']
    
    for k in range(1, ke):
        ex[k] = ex[k] + courant_number * (hy[k - 1] - hy[k])

    pulse = exp(-courant_number * ((t0 - frame) / spread) ** 2)
    ex[kc] = pulse

    for k in range(ke - 1):
        hy[k] = hy[k] + courant_number * (ex[k] - ex[k + 1])

    line_ex.set_ydata(ex)
    line_hy.set_ydata(hy)

    return line_ex, line_hy

def run_animation(params):

    ke = params['ke']
    kc = params['kc']
    ex = params['ex']
    hy = params['hy']
    nsteps = params['nsteps']

    fig, (ax_ex, ax_hy) = plt.subplots(2, 1)
    
    ax_ex.set_ylabel('E$_x$')
    ax_ex.set_xlabel('FDTD cells')
    ax_ex.set_ylim(-1.2, 1.2)
    ax_ex.set_xlim(0, ke)
    ax_ex.axvline(x=kc, color='r', linestyle='--', label='Source Location')
    ax_hy.set_ylabel('H$_y$')
    ax_hy.set_xlabel('FDTD cells')
    ax_hy.set_ylim(-1.2, 1.2)
    ax_hy.set_xlim(0, ke)
    ax_hy.axvline(x=kc, color='r', linestyle='--', label='Source Location')

    global line_ex, line_hy
    line_ex, = ax_ex.plot(range(ke), ex, color='k', linewidth=1)
    line_hy, = ax_hy.plot(range(ke), hy, color='k', linewidth=1)

    start_time = time.time()
    ani = FuncAnimation(fig, animate, frames=range(1, nsteps+1), fargs=(params,), init_func=lambda: init(params), blit=True, interval=50)

    execution_time = time.time() - start_time
    print(f"execution time = {execution_time} seconds")
    
    plt.tight_layout()
    path_save = os.path.join(params['path_save'], f'animation_{ke}.gif')
    ani.save(filename=path_save, writer="pillow")

