import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math, os
from glob import glob

from .logger import Logger


def plot_MIAs(
    logs,
    key,
    color,
    label,
    xlimcut=None,
    ax=None,
    linestyle='-',
    marker='x',
    alpha_std=.1,
    x_axis_gen_error=True,
):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    iterations = logs['iteration'][0]
    gen_error = get_gen_error(logs) 
    
    if x_axis_gen_error:
        x = gen_error
    else:
        x = iterations
        
    
    if xlimcut:
        lim = np.sum(x <= xlimcut) + 1
    else:
        lim = -1
        
    Y = logs[key][0][0:lim] - .5
    Yvar = logs[key][1][0:lim]
    x = x[0:lim]

    perm = np.argsort(x)
    Y = Y[perm]
    Yvar = Yvar[perm]
    x = x[perm]
    
    ax.fill_between(x, (Y-Yvar), (Y+Yvar), color=color, alpha=alpha_std)
    ax.plot(x, Y, color=color, label=label, marker=marker, linestyle=linestyle)
        
    return ax, lim

def plot_consesus(logs, ax, xcut):
    """ Adds consensus dist. curve to a plot """
    mn, mx = ax.get_ylim()
    
    ax2 = ax.twinx()

    ax2.set_ylabel('Consensus distance ($C$)', color='black')
    
    gen_error = get_gen_error(logs)[:xcut]
    perm = np.argsort(gen_error)

    consensus = logs['distance'][0][:xcut]
    consensus = consensus[perm]
    consensus_var = logs['distance'][1]
    consensus_var = consensus_var[perm]
    
    ax2.plot(gen_error, consensus, color='black', label="Consensus distance", alpha=1, linestyle='--')
    ax2.ticklabel_format(axis='y', style="sci", scilimits=(0,0));

def plot_utility(ax, logs, label, color):
    if ax is None:
        fig, ax = plt.subplots(1,1)

    x = logs['iteration'][0]
    ax.plot(x, logs['accuracy_on_test'][0], label=f"{label}-[test]", color=color, marker='x')
    ax.plot(x, logs['accuracy_on_train'][0], label=f"{label}-[train]", color=color, linestyle='--', marker='x', alpha=.5)
    ax.legend();
    return ax

def get_gen_error(logs):
    """ Computes generalization error on aggregated logs """
    acc_test = logs['accuracy_on_test'][0]
    acc_train = logs['accuracy_on_train'][0]
    gen_error = (acc_train - acc_test) / (acc_test + acc_train)  
    return gen_error

def load_parse_aggregated_logs(home, parser, metric):
    """ Util function to load, parse and aggregate (compute avg.) log files """

    paths = glob(home)
    print(f'Aggregating: {home}. Results averaged over {len(paths)} runs.')
    
    # get average length runs
    ns = np.array([len(Logger.load(path)[1]['iteration']) for path in paths])
    n = math.floor(ns.mean())
    
    # parse files
    avg_logs = {}
    for key, f in parser.items():
        # all work done here
        avg_logs[key] = aggregate(key, paths, n, f, metric)
        
    return avg_logs


def aggregate(key, paths, n, f, metric):
    attributes = []
    for i, path in enumerate(paths):

        # read file
        conf, logs = Logger.load(path)   

        # get attribute
        attribute_full = logs[key]
        # parse attribute
        attribute_full = f(conf, logs, attribute_full, metric)
        
        # pad or cut bassed on avg length
        if len(attribute_full) >= n:
            attribute = attribute_full[:n]
        elif len(attribute_full) < n:
            attribute = np.full(n, np.nan)
            attribute[:len(attribute_full)] = attribute_full

        attributes.append(attribute)
    
    attributes = np.array(attributes)
    avg_attributes = np.nanmean(attributes, 0)
    std_attributes = np.nanstd(attributes, 0)
    
    return avg_attributes, std_attributes

    
## Parsers attributes
    
def parser_avg_on_neighbors(conf, logs, mia, metric, attacker=0):
    """ Compute attibute avg only on the attacker's neighbors """
    G = conf['att_neighbors']
    
    out = []
    for j in range(len(mia)):
        y = np.array(mia[j][:, metric])
        ymean = y.mean()
        out.append(ymean)
    mia_mean = np.array(out)
    return mia_mean

parser_avg_all = lambda conf, logs, data, metric: np.array([np.array(m).mean() for m in data])

parse_avg_id = lambda conf, logs, data, _: data
