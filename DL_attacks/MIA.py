import tensorflow as tf
import numpy as np

from .utils import *
from .user import *
import functools

def compute_modified_entropy(p, y, epsilon=0.00001):
    """ Computes label informed entropy from 'Systematic evaluation of privacy risks of machine learning models' USENIX21 """
    assert len(y) == len(p)
    n = len(p)

    entropy = np.zeros(n)

    for i in range(n):
        pi = p[i]
        yi = y[i]
        for j, pij in enumerate(pi):
            if j == yi:
                # right class
                entropy[i] -= (1-pij)*np.log(pij+epsilon)
            else:
                entropy[i] -= (pij)*np.log(1-pij+epsilon)

    return entropy


def ths_searching_space(nt, train, test):
    """ it defines the threshold searching space as nt points between the max and min value for the given metrics """
    thrs = np.linspace(
        min(train.min(), test.min()),
        max(train.max(), test.max()), 
        nt
    )
    return thrs

def mia_best_th(model, train_set, dl, nt=150):
    """ Perfom naive, metric-based MIA with 'optimal' threshold """
    
    def search_th(Etrain, Etest):
        R = np.empty(len(thrs))
        for i, th in enumerate(thrs):
            tp = (Etrain < th).sum()
            tn = (Etest >= th).sum()
            acc = (tp + tn) / (Etrain.shape[0] + Etest.shape[0])
            R[i] = acc
        return R.max()
    
    # evaluating model on train and test set
    Ltrain, _, Ptrain, Ytrain = dl.attacker.evaluate(train_set, model=model)
    Ltest, _, Ptest, Ytest = dl.attacker.evaluate(dl.test_set, model=model)
    
    # it takes a subset of results on test set with size equal to the one of the training test 
    n = Ptrain.shape[0]
    Ptest = Ptest[:n]
    Ytest = Ytest[:n]
    Ltest = Ltest[:n]
        
    # performs optimal threshold for loss-based MIA 
    thrs = ths_searching_space(nt, Ltrain, Ltest)
    loss_mia = search_th(Ltrain, Ltest)
    
    # computes entropy
    Etrain = compute_modified_entropy(Ptrain, Ytrain)
    Etest = compute_modified_entropy(Ptest, Ytest)
    
    # performs optimal threshold for entropy-based MIA 
    thrs = ths_searching_space(nt, Etrain, Etest)
    ent_mia = search_th(Etrain, Etest)
    
    return loss_mia, ent_mia


def mia_for_each_nn(dl, attacker, get_model):
    """ Run MIA for each attacker's neighbors """
    
    nn = sorted(attacker.neighbors, key=lambda x:int(x.name))
    model = deepCopyModel(attacker.model)

    mias = np.zeros((len(nn), 2))
    for i, v in enumerate(nn):
        var = get_model(attacker, v)
        
        assign_list_variables(model.trainable_variables, var)
        
        train_set = v.train_set
        
        mias[i] = mia_best_th(model, train_set, dl)
        
    return mias


def MIA_local_model(
    dl,
    attacker,
):
    get_model = lambda attacker, u: attacker.model.trainable_variables
    
    return mia_for_each_nn(dl, attacker, get_model)


def MIA_received_model(
    dl,
    attacker,
):
    get_model = lambda attacker, u: attacker.model_update_buffer[u.name]
    
    return mia_for_each_nn(dl, attacker, get_model)


def MIA_funcIsolated_model(
    dl,
    attacker,
):
    get_model = lambda attacker, u: attacker.get_functional_isolated_model(u.name)
    
    return mia_for_each_nn(dl, attacker, get_model)







