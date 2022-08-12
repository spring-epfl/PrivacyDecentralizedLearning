import numpy as np
import tensorflow as tf

from .user import *
from .DL import DecentralizedLearning


class User_FedAVG(User):
    ...

class Attacker_FedAVG0(User):
        
    def train(self):
        ...
        
    def update(self):
        self.iter += 1

class FederatedAVG(DecentralizedLearning):
                    
    def setup(
        self,
        n_users,
        make_model,
        train_sets,
        test_set,
        user,
        attacker
    ):
        
        DecentralizedLearning.setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            user,
            attacker
        )
            
        for i in range(1, self.n_users):
            self.attacker.neighbors.add(self.U[i])
            
        self.test_set = test_set
             
        
    def __call__(self):
        # get model updates
        new_model = init_list_variables(self.U[0].model.trainable_variables)
        
        # compute AVG
        for u in self.U:
            u.train()            
            u_model = u.get_model_update(u)
            new_model = agg_sum(new_model, u_model)
            
        new_model = agg_div(new_model, self.n_users)
        
        # send models
        for u in self.U:
             assign_list_variables(u.model.trainable_variables, new_model)


    