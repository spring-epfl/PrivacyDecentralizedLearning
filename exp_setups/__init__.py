import numpy as np
from DL_attacks import model, user, attacker, attacker_active, DL
import random

# where to save logs
output_dir = './results'

# Graph topology
CDL = DL.DecentralizedLearning
USER = user.User
ATTACKER = attacker.Attacker
G = None

# learning-rate scheduler steps to reach consensus (it may vary based on the topology)  
lrd = [300, 400, 500]
    
# maximum number of training iterations 
max_num_iter = 1000
# attacker node
ATTACKER_ID = 0
# additional conf for topology
graph_properties = {}
# is it an active attack?
active = False

# initial learning rate
init_lr = .1

# patience early stopping
patience = 3
# when to run MIAs
eval_interval = 25
# is it federated learning?
federated = False
        

## Obsolete #############################
# nodes starts with the same parameters 
model_same_init = True
# how to create local training sets [0: uniform]
type_partition = 0