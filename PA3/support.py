import numpy as np
import os
import yaml
import csv
from scipy.constants import c
from math import ceil
import matplotlib.pyplot as plt


def plot_output_image(params, ex, hy, time_step=None):

    kc = params['kc']
    ke = params['ke']

    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(8, 3.5))

    plt.subplot(211)
    plt.plot(ex, color='k', linewidth=1)
    plt.ylabel('E$_x$', fontsize='14')
    plt.xticks(np.arange(0, ke+1, step=max(20, ke//10)))
    plt.xlim(0, params['ke'])
    plt.yticks(np.arange(-1, 1.2, step=1))
    plt.ylim(-1.2, 1.2)
    plt.axvline(x=kc, color='r', linestyle='--', label='Source')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(hy, color='k', linewidth=1)
    plt.ylabel('H$_y$', fontsize='14')
    plt.xlabel('FDTD cells')
    plt.xticks(np.arange(0, ke+1, step=max(20, ke//10)))
    plt.xlim(0, params['ke'])
    plt.yticks(np.arange(-1, 1.2, step=1))
    plt.ylim(-1.2, 1.2)
    plt.axvline(x=kc, color='r', linestyle='--', label='Source')
    plt.legend()

    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    
    path_save = os.path.join(params['path_save'], f'output_{ke}.png')
    plt.savefig(fname=path_save)


def write_stats(params, execution_time):

    device = params['device']

    path_write = os.path.join(params['path_stats'], f'stats-{device}.csv')
    file_exists = os.path.isfile(path_write)

    with open(path_write, mode='a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Extent', 'Execution Time'])
        writer.writerow([params['ke'],execution_time])

def update_params(params):

    params['ex'] = np.zeros(params['ke'])
    params['hy'] = np.zeros(params['ke'])
    
    # pulse params
    params['kc'] = int(params['ke'] / 2)
    params['dt'] = params['kc'] / (2 * c) 
    params['courant_number'] = c * params['dt'] / params['kc']

    params['nsteps'] = params['ke']
    return params
 
def parse_args(all_args):

    tags = ["--", "-"]

    all_args = all_args[1:]

    if len(all_args) % 2 != 0:
        print("Argument '%s' not defined" % all_args[-1])
        exit()

    results = {}

    i = 0
    while i < len(all_args) - 1:
        arg = all_args[i].lower()
        for current_tag in tags:
            if current_tag in arg:
                arg = arg.replace(current_tag, "")
        results[arg] = all_args[i + 1]
        i += 2

    return results

def load_yaml(argument):

    return yaml.load(open(argument), Loader=yaml.FullLoader)

def load_config(sys_args):

    args = parse_args(sys_args)

    return load_yaml(args['config'])

