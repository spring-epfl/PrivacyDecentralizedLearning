import os, sys, importlib
import tensorflow as tf
import numpy as np
import tqdm

from DL_attacks.utils import EarlyStopping, setup_data, setup_model
from DL_attacks.logger import Logger

if __name__ == '__main__':
    
    try:
        ds_setup_file = sys.argv[1]
        top_setup_file = sys.argv[2]
        run_num = sys.argv[3]
    except:
        print(f"USAGE: dataset_setup_file topology_setup_file run_number")
        sys.exit(1)
        
    
    Cds = importlib.import_module(ds_setup_file)
    Ctop = importlib.import_module(top_setup_file)

    name = f'{run_num}-{Cds.dsk}-{Ctop.name}'
    output_file = os.path.join(Cds.output_dir, name)
    print(f"Logging file in --> {output_file}")
    
    # gets users' local training size
    size_local_ds = Cds.compute_local_training_set_size(Ctop.nu)
    
    print("Running setup ....")
    # loads and splits local training sets and test one (validation)
    train_sets, test_set, x_shape, num_class = setup_data(
        Cds.load_dataset,
        Ctop.nu,
        size_local_ds,
        Cds.batch_size,
        Cds.size_testset,
        Cds.type_partition
    )

    # setup model generator function
    make_model = setup_model(
        Cds.model_maker,
        [x_shape, num_class, Ctop.init_lr, Ctop.lrd],
        Cds.model_same_init
    )
    
    # define comm. topology
    DL = Ctop.CDL(Ctop.graph_properties)
    if Ctop.G is None:
        DL.setup(Ctop.nu, make_model, train_sets, test_set, Ctop.USER, Ctop.ATTACKER)
    else:
        DL.from_nx_graph(Ctop.G, make_model, train_sets, test_set, Ctop.USER, Ctop.ATTACKER)

    # it runs and logs metric during the training, including privacy risk
    logr = Logger(Ctop, DL, output_file)
    # it implements early stopping
    es = EarlyStopping(Cds.patience)
    
    ## Main training loop
    print("Training ....")
    for i in range(1, Cds.max_num_iter+1):
        # run a round of DL
        DL()
        
        # eval models
        if i % Cds.eval_interval == 0 and i:
            # logs privacy risk (slow operation)
            score = logr(i)
            
            # checks for early stopping
            if es(i, score):
                print("\tEarly stop!")
                break
            
            # save current logs
            logr.dump()
    
    # final evaluation
    logr(i, DL)
    
    # save final logs
    logr.dump()