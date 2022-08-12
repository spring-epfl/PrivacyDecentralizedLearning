# On the Privacy of Decentralized Machine Learning

Code for the paper:

*[On the Privacy of Decentralized Machine Learning](https://arxiv.org/abs/2205.08443)*, Dario Pasquini, Mathilde Raynal, and Carmela Troncoso

## Requirements 

* TensorFlow2.x (tested on tf2.6 and tf2.9)
* [networkx](https://networkx.org)
* [tqdm](https://tqdm.github.io/docs/tqdm/)
* numpy, matplotlib

## Main modules

The most important modules in the repository are:

* *DL_attacks/DL.py*: it implements decentralized learning.
* *DL_attacks/user.py*: it implements the decentralized user functionality, including local training and local parameters aggregation.
  * *DL_attacks/attacker.py*: it builds on top of *user.py* and implements passive adversarial users.
* *DL_attacks/logger.py*: it is a support module that takes care of logging various metrics  during the decentralized learning protocol, including privacy risk. 
* *DL_attacks/MIA.py*: it implements a set of simple membership inference attacks and it is used in *logger.py* to measure privacy risk.

* *run_experiment.py*: it is the main where training and logging are performed. It takes as input some configuration files (more on this later) and writes on disk all the collected information.

## How do configuration files work?

Configuration files must be placed in *exp_setups* and every configuration file is a python module. There are two main types of configuration files:

* *"Dataset setting files"* that define which dataset to use during the training, how to split it among users, and which model to use (but also minor things such as dataset processing, batch size, etc.) e.g., *exp_setups/CIFAR10.py*.
* *"Topology setting files"* that define which topology to use during the training and related configurations e.g., *exp_setups/torus36.py*.

Additionally, *exp_setups/\_\_init\_\_.py* contains a list of additional configurable parameters. *exp_setups/\_\_init\_\_.py* should be always included in all the configuration files. 

## How to run?

As we said, *run_experiment.py* is the main. This takes as input:

* a *Dataset setting file*.
* a *Topology setting file*.
* a *Run id*---an integer that identifies the run once written on disk.

For instance,

```
py run_experiment.py exp_setups.CIFAR10 exp_setups.torus36 0
```

runs decentralized learning on CIFAR10 with a torus topology (36 users). At the end of the training, it should write in *./results* a file named *0-cifar10-torus36* that contains all the logged information. Results can be then visualized via the notebook *plot_privacy_attacks_passive.ipynb*. The notebook aggregates multiple runs and plots averaged results.

Another example is

```
py run_experiment.py exp_setups.CIFAR10 exp_setups.federated36 0
```

that does the same for federated learning.

Note that the code simulates a decentralized protocol sequentially i.e., only a GPU is used per run.

To run multiple *"run_experiment.py"* in parallel, you can use *"multiGPU_run_experiment.py"*. Assuming a machine with multiple GPUs, it runs multiple sessions of *"run_experiment.py"* assigning different GPUs to processes (nvidia only). 

This takes as input:

* *Dataset setting file*
* *Topology setting file*
* How many runs e.g., 16
* How many GPUs (parallel runs) to cast e.g., 4
* DEVICE-ID of the first GPU to use e.g., 0

For instance,

```
py multiGPU_run_experiment.py exp_setups.CIFAR10 exp_setups.torus36 16 4 0
```

runs a total of 16 experiments (CIFAR10 on torus36) using 4 GPUs. As for *run_experiment.py*, results are written in *./results*.

## Other attacks

* *'State-override_with_active_grad_inv.ipynb'* reports an end-to-end example of active gradient inversion attack via the state-override attack.

## What's missing?

* Configuration files and code for other attacks (e.g., gradient recovery, echo, and state-override) will be online soon.





