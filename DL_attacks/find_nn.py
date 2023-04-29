import tensorflow as tf
import numpy as np
import tqdm
import math
import itertools

from DL_attacks.ops_on_vars_list import *


def find_nn(dl, victim, number_of_neighbors_range, lr=1.):
    adv = dl.attacker
    victim = victim.name
    # get model update sent by the victim at the current round
    target_w = flat_tensor_list(adv.window_model_update_buffer[1][victim])
    
    # get all attacker' model updates received at the previous round
    mups = dl.attacker.window_model_update_buffer[0]
    keys = sorted(mups.keys())
    # flat them and make a matrix with all the model updates
    mups = np.array([flat_tensor_list(mups[k]) for k in keys])
    

    nodes = list(range(len(keys)))
    # remove trivial elements (adv and victim are always neighbor of the victim)
    nodes.remove(victim)
    nodes.remove(adv.name)

    scores = []
    i = 0
    # brute-force
    for n in number_of_neighbors_range:
        for comb in itertools.combinations(nodes, n):
            # add trivial neighbors
            comb = list(comb) + [victim, adv.name]
            comb = sorted(comb)
            
            # gather previous model updates
            model = mups[comb,:].sum(0) / len(comb)
            # attempt gradient recovery
            gradient = target_w - model
            gradient = gradient/lr
            
            # attempt model recovery
            pseudo_target_w = model + gradient
            # compute difference recovered and received
            diff = target_w - pseudo_target_w
            diff = np.abs(diff).sum()
            scores.append( (diff, comb) )
            i += 1
    # sort by difference (the first element should be the right one)
    s_scores = sorted(scores, key=lambda x: x[0])

    
    # Check result inference
    ## get ground truth
    id_nn = [v.name for v in dl.U[victim].neighbors]
    id_nn += [victim]
    ground_truth = sorted(id_nn)
    
    ## get rank ground truth according infered  
    for solution_rank, (_, elements) in enumerate(s_scores):
        if elements == ground_truth:
            break
            
    return solution_rank, ground_truth, s_scores