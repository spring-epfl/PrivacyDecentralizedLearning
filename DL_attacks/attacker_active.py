from .user import *
from .attacker import *


class Echo_received_model(Attacker):
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.accumulated_models = {}
        self.vitcim = None
        self.counter = 0
    
    def train(self):
        # if victim is not chosen, pick the neighbor with lowest id
        if self.vitcim is None:
            nn = [v.name for v in self.neighbors]
            nn = sorted(nn)
            self.vitcim = [nn[0]]
            
        assign_list_variables(self.model.trainable_variables, self.model_update_buffer[self.vitcim[0]])
        
    def update(self):
        self.iter += 1
        
    def get_functional_isolated_model(self, victim_id, marg_type=0):
        buffer = select_nn_mus(self.neighbors, self.model_update_buffer)
        var = isolate_victim(buffer, victim_id)
        return var
     
    
class Echo_isolated_model(Echo_received_model):
        
    def train(self):        
        if self.vitcim is None:
            nn = [v.name for v in self.neighbors]
            nn = sorted(nn)
            self.vitcim = [nn[0]]

        buffer = select_nn_mus(self.neighbors, self.model_update_buffer)            
        var = isolate_victim(buffer, self.vitcim[0])
        assign_list_variables(self.model.trainable_variables, var)
        
    def get_functional_isolated_model(self, victim_id, marg_type=0):
        var = self.model.trainable_variables
        return var

                
class StateOverideAttacker(User):

    def make_payload(self):
        raise Exception('Not Implemented')
        
    def get_targets(self):
        raise Exception('Not Implemented')
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.victim, self.victim_nn = self.get_targets()
        
        self.payload = self.make_payload()
        

    def get_model_update(self, victim):
        
        if not victim.name == self.victim:
            return User.get_model_update(self, victim)
              
        adv_theta = clone_list_tensors(self.model_update_buffer[victim.name])
        for nn in self.victim_nn:
            adv_theta = agg_sum(adv_theta, self.model_update_buffer[nn])
            
        payload = self.payload
        payload = agg_div(payload, 1/(len(self.victim_nn) + 1 + 1))
        
        adv_theta = agg_neg(adv_theta)
        adv_theta = agg_sum(adv_theta, payload)
        return adv_theta
                
