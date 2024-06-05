import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from support import load_config

plt.style.use("ggplot")

def load_csv(params):

    device = params['device']

    csv_path = os.path.join(params['path_stats'],f'stats-{device}.csv')
    df = pd.read_csv(csv_path)

    return df

def plot_timing_stat_single(params):

    device = params['device']
    df = load_csv(params)

    fig, ax = plt.subplots()

    ax.plot(df['Extent'], df['Execution Time'], marker='o', linestyle='-')
    ax.set_xlabel('Spatial Extent (arbitrary units)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Spatial Extent vs. Execution Time')
    
    path_save = os.path.join(params['path_save'],f'timing-{device}.png')
    plt.savefig(fname=path_save)

def plot_timing_stats(gpu_stats, cpu_stats):

    plt.figure(figsize=(10, 6))

    plt.plot(cpu_stats['Extent'], cpu_stats['Execution Time'], marker='o', label='CPU')
    plt.plot(gpu_stats['Extent'], gpu_stats['Execution Time'], marker='s', label='GPU')
    
    plt.xlabel('Extent (log scale)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Comparison of CPU and GPU Execution Times')
    plt.legend()
    plt.grid(True)
    
    plt.xscale('log') 
    
    #plt.show()
    #plt.tight_layout()
    path_save = os.path.join(params['path_save'],f'timing.png')
    plt.savefig(fname=path_save)

if __name__=="__main__":

    params = load_config(sys.argv)

    params['device'] = 'gpu'
    gpu_df = load_csv(params)

    params['device'] = 'cpu'
    cpu_df = load_csv(params)
   
     
    plot_timing_stats(gpu_df, cpu_df) 
