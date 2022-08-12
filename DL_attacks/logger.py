import os, pickle

from . import MIA

class Logger:
    """ It runs and logs metric during the training, including privacy risk for models """
    
    def __init__(self, Ctop, DL, output_path):
        
        self.Ctop = Ctop
        self.DL = DL
        self.output_path = output_path

        # init log structure
        self.info = {}
        if Ctop.nu > 1:
            DL.model_graph();

            att_neighbors = list(DL.G.neighbors(DL.attacker.name))

            self.info = {
                'att_neighbors' : att_neighbors,
                'graph' : DL.G,
            }
        
        self.logs = {
            'distance' : [],

            'accuracy_on_test':[],
            'loss_on_test':[],

            'accuracy_on_train':[],
            'loss_on_train':[],

            'MIA_local_model':[],
            'MIA_received_model':[],
            'MIA_isolated_model':[],

            'iteration':[],
        }
        
        
    def __call__(self, j, verbose=True):
        # this is the function that does all the work---it computes and logs utility, consensus and privacy 
        
        ATTACKER_ID = self.Ctop.ATTACKER_ID
        attacker = self.DL.U[ATTACKER_ID]
        
        self.logs['iteration'].append(j)
        
        (loss_train, acc_train), (loss_test, acc_test) = self.DL.train_test_utility(drop_attacker=self.Ctop.active)
        consenus_distance = self.DL.compute_models_distance()
        
        self.logs['accuracy_on_test'].append(acc_test)  
        self.logs['loss_on_test'].append(loss_test)  
        
        self.logs['accuracy_on_train'].append(acc_train) 
        self.logs['loss_on_train'].append(loss_train)  
        
        self.logs['distance'].append(consenus_distance) 

        # check privacy risk
        if not self.Ctop.federated:
            mia_score = MIA.MIA_received_model(self.DL, attacker)
            self.logs['MIA_received_model'].append(mia_score)

            mia_score = MIA.MIA_funcIsolated_model(self.DL, attacker)
            self.logs['MIA_isolated_model'].append(mia_score)
            
        else:
            mia_score = MIA.MIA_local_model(self.DL, attacker)
            self.logs['MIA_local_model'].append(mia_score)
            
        if verbose:
            gen_error = acc_train - acc_test
            log_msg = f'[ROUND: {j:05d}] Acc_val: {acc_test:.3f}, Acc_train: {acc_train:.3f}, Gen. error: {gen_error:.3f}, Consenus dist. {consenus_distance.mean():.3e}'
            print(log_msg)
            
        return acc_test
    
    
    def dump(self):
        if not os.path.isdir(self.Ctop.output_dir):
            os.mkdir(self.Ctop.output_dir)
        
        with open(self.output_path, 'wb') as f:
            pickle.dump((self.info, self.logs), f)
            
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            logs = pickle.load(f)
        return logs