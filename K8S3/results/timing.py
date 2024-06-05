import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from support import load_config

plt.style.use("ggplot")

def load_csv(params):

    csv_path = os.path.join(params['path_stats'],'stats.csv')
    df = pd.read_csv(csv_path)

    return df

def plot_timing_stats(params):

    df = load_csv(params)

    fig, ax = plt.subplots()

    ax.plot(df['Extent'], df['Execution Time'], marker='o', linestyle='-')
    ax.set_xlabel('Spatial Extent (arbitrary units)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Spatial Extent vs. Execution Time')
    
    path_save = os.path.join(params['path_save'],'timing.png')
    plt.savefig(fname=path_save)

if __name__=="__main__":

    params = load_config(sys.argv)
   
    plot_timing_stats(params) 
