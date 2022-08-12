import numpy as np
import networkx as nx
import math, random

from .utils import *
from .user import *

class DecentralizedLearning:
    
    def __init__(
        self,
        graph_properties
    ):
        self.graph_properties = graph_properties
        self.attacker = None
        self.U = None
        self.test_set = None
        self.n_users = None
        
        self.global_model = None
        
    def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            user,
            attacker,
    ):
        
        assert len(train_sets) == n_users
        
        self.U = [None] * n_users
        self.n_users = n_users
        self.test_set = test_set
        
        # attacker is always the first user
        self.attacker = attacker(0, make_model, train_sets[0])
        self.U[0] = self.attacker
        
        for i in range(1, self.n_users):
            self.U[i] = user(i, make_model, train_sets[i])    

    
    def from_nx_graph(
        self,
        G,
        make_model,
        train_sets,
        test_set,
        user,
        attacker,
        shuffle=True
    ):
        """ Comm. topology from networkx graph """
        
        nodes = list(G.nodes)
        DecentralizedLearning.setup(
            self,
            len(nodes),
            make_model,
            train_sets,
            test_set,
            user,
            attacker
        )
        
        mmap = {}
        if shuffle:
            random.shuffle(nodes)
        
        for i in range(self.n_users):
            u = nodes[i]
            mmap[u] = i
                  
        for u, v in G.edges:
            u = mmap[u]
            v = mmap[v]
            self.U[u].neighbors.add(self.U[v])
            self.U[v].neighbors.add(self.U[u])      
           

    def __call__(self):
        """ Exucutes one round: local training + communication + aggregation for every user """
        
        # get model updates
        pool = self.U[1:]
        for u in pool:
            u.train()            
            mu = u.get_model_update(u)
            u.model_update_buffer[u.name] = mu
            u.check_model(u.name, mu)
            
            # send the local model update to neighbors
            for v in u.neighbors:
                mu = u.get_model_update(v)
                v.check_model(u.name, mu)
                v.model_update_buffer[u.name] = mu
                    
                    
        # attacker acts after everyone else (only for active attacks, when needed)
        self.attacker.train()  
        mu = self.attacker.get_model_update(self.attacker)
        self.attacker.model_update_buffer[self.attacker.name] = mu
        for v in self.attacker.neighbors:
            mu = self.attacker.get_model_update(v)
            v.check_model(self.attacker.name, mu)
            v.model_update_buffer[self.attacker.name] = mu
                    
        pool = self.U
        for u in pool:
            u.check_models()
            u.update()
    
       
    def compute_global_model(self, drop_attacker=False):
        """ Computes the global model i.e., the average of all local models (but the attacker on the malicious model)"""
        
        # init arch
        if self.global_model is None:
            self.global_model = deepCopyModel(self.attacker.model)
            
        # init weights to 0s
        global_vars = init_list_variables(self.attacker.model.trainable_variables)
            
        # do not consider the attaker when active
        if drop_attacker:
            users = self.U[1:]
        else:
            users = self.U
        
        # avg local models
        for u in users:
            global_vars = agg_sum(global_vars, u.model.trainable_variables)
        global_vars = agg_div(global_vars, len(users))
        
        assign_list_variables(self.global_model.trainable_variables, global_vars)
        
        return self.global_model
    
    
    def train_test_utility(self, drop_attacker=False):
        
        self.compute_global_model(drop_attacker=drop_attacker)
        
        if drop_attacker:
            users = self.U[1:]
        else:
            users = self.U
        
        loss_train, acc_train = 0., 0.
        for u in users:
            _loss_train, _acc_train = u.evaluate(u.train_set, model=self.global_model)[:2]
            loss_train += _loss_train.mean()
            acc_train += _acc_train.mean()

        loss_train = loss_train / len(users)
        acc_train = acc_train / len(users)
        
        loss_test, acc_test = self.U[0].evaluate(self.test_set, model=self.global_model)[:2]
        loss_test = loss_test.mean()
        acc_test = acc_test.mean()
        
        return (loss_train, acc_train), (loss_test, acc_test)
    
    
    def model_graph(self, with_labels=True, node_color=None):
       
        self.G = nx.Graph()
        for i in range(self.n_users):
            self.G.add_node(self.U[i].name)
            
        for u in self.U:
            for v in u.neighbors: 
                e = (u.name, v.name) if u.name > v.name else (v.name, u.name)
                self.G.add_edge(*e)
                
        if node_color is None:
            node_color = [[0, 1, 0]] * self.n_users
            # attacker
            node_color[0] = [1, 0, 0]
            # attacker's neighbors
            for v in self.G.neighbors(self.attacker.name):
                node_color[int(v)] = [1, .7, .8]
                      
        nx.draw(self.G, with_labels=with_labels, node_color=node_color);
        
        
    def compute_models_distance(self):
        """ Computes the consensus distance """
        m = flat_tensor_list(self.U[0].model.trainable_variables).shape[0]
        
        params = np.zeros((self.n_users,m))
        for i, u in enumerate(self.U):
            params[i] = flat_tensor_list(u.model.trainable_variables)

        n = params.shape[0]
        dmatrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dmatrix[i,j] = ((params[i] - params[j]) ** 2).mean()
        return dmatrix        
    
#Pre-defined top. -------------------------------------------------------------------------------------
    
class Ring(DecentralizedLearning):   
    def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            user,
            attacker,
    ):

        G = nx.random_regular_graph(2, n_users, seed=0)
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, user, attacker, shuffle=False)
                            

class Torus(DecentralizedLearning):
     def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            user,
            attacker,
    ):
        
        """ Torus comm. topology. n_users must have a square root """
        n = math.sqrt(n_users)
        assert n % 1 == 0
        
        n = int(n)
        G = nx.grid_graph(dim =[n, n], periodic=True)
        
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, user, attacker, shuffle=False)
            

                