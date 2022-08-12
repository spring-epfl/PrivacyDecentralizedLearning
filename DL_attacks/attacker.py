import tensorflow as tf
import numpy as np

from .utils import *
from .user import *
from .ops_on_vars_list import *

def isolate_victim(model_update_buffer, victim_id):
    """ Computes marginalized model  """
    
    # Received model updates for the round
    thetas = model_update_buffer.copy()
    n = len(thetas)

    # Remove victim's one
    victim = thetas.pop(victim_id)
    others = thetas
    
    # accumulate others
    other = init_list_variables(victim)
    for _, var in others.items():
        other = agg_sum(other, var)
    other = agg_div(other, n)

    # remove global functionality
    victim_c = agg_sub(victim, other)
    # scale back marginalized model
    victim_c = agg_div(victim_c, 1/n)
    
    return victim_c

            
class Attacker(User):
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.vitcim = self.neighbors
        
    def get_functional_isolated_model(self, victim_id):
        var = isolate_victim(self.model_update_buffer, victim_id)
        return var
    
    def __repr__(self):
        return "Attacker: " + str(self.name)