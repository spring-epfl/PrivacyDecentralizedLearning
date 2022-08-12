import tensorflow as tf
import numpy as np

from .ops_on_vars_list import *

class User:

    def __init__(
        self,
        name,
        make_model,
        train_set,
    ):
        self.name = name
        self.train_set = train_set
        self.train_set_iter = iter(self.train_set.repeat(-1))
        self.neighbors = set()
        self.model, self.loss, self.opt, self.metric = make_model()
        self.opt = self.opt()
        
        # received model updates at the current round
        self.model_update_buffer = {}
        
        self.window_model_update_buffer = [None, None]
        self.window_local_model = [None, None]
        self.window_training_data = [None, None]
        self.window_gradient = [None, None]
        self.gradient = None 

        self.history = []
        self.iter = 0
        
        
    def get_model_update(self, user):
        """ Generate model update for user 'user' """
        var = self.model.trainable_variables
        var = clone_list_tensors(var)
        return var
    
    
    def compute_loss(self, x, y, model, training=True):
        p = model(x, training=training)
        loss = self.loss(y, p)
        return p, loss
        
        
    def train(self):
        """ Local training step """
        
        # get data
        x, y = next(self.train_set_iter)

        with tf.GradientTape() as tape:
            p, loss = self.compute_loss(x, y, self.model, training=True)
            loss = tf.reduce_mean(loss)
            metric = self.metric(y, p)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(g, self.model.trainable_variables))
        
        # logging 
        self.gradient = g

        out = (loss.numpy(), metric.numpy().mean())
        self.history.append(out)
        
        self.window_gradient[0] = self.window_gradient[1] 
        self.window_gradient[1] = clone_list_tensors(g)
        
        self.window_training_data[0] = self.window_training_data[1] 
        self.window_training_data[1] = x.numpy()
        ##
        
        
    def update(self):
        """ Update state based on received model updates (and local) """
        
        nups = len(self.model_update_buffer)
        new_theta = init_list_variables(self.model.trainable_variables)

        for theta in self.model_update_buffer.values():
            new_theta = agg_sum(new_theta, theta)
            
        new_theta = agg_div(new_theta, nups)
                    
        # logging
        if self.window_model_update_buffer[1] is None:
            self.window_model_update_buffer[0] = None
        else:
            self.window_model_update_buffer[0] = self.window_model_update_buffer[1].copy()
        self.window_model_update_buffer[1] = self.model_update_buffer.copy()
        ##
        
        # set new params for local model
        assign_list_variables(self.model.trainable_variables, new_theta)
        
        # logging
        self.window_local_model[0] = self.window_local_model[1]
        self.window_local_model[1] = clone_list_tensors(self.model.trainable_variables)
        ## 
        self.iter += 1

    def check_model(self, user, var):
        """ hook function: called when the user receive the model update from user ‘user’ """
        ...
        
    def check_models(self):
        """ hook function: called when all the model update have been received (before update()) """
        ...

    def evaluate(self, dataset, model=None, take=-1):
        loss = []
        metric = []
        output = []
        Y = []
        
        if model is None:
            model = self.model
        
        for x, y in dataset.take(take): 
            p, _loss = self.compute_loss(x, y, model)
            _metric = self.metric(y, p)
                        
            loss.append(_loss.numpy())
            metric.append(_metric.numpy())
            output.append(p.numpy())
            Y.append(y)
               
        loss = np.concatenate(loss)
        metric = np.concatenate(metric)
        output = np.concatenate(output)
        Y = np.concatenate(Y)
        
        return loss, metric, output, Y
      
    def __repr__(self):
        return "User: " + str(self.name)