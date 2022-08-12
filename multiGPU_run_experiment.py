import os, sys, glob
from tqdm import tqdm
from DL_attacks.utils import runMultiGPU

EX = "python run_experiment.py %s %s %s"

try:
    ds_setup_file = sys.argv[1]
    top_setup_file = sys.argv[2]
    num_run = int(sys.argv[3])
    nGPU = int(sys.argv[4])
    shift = int(sys.argv[5])
except:
    print("USAGE: 'dataset_setup_file topology_setup_file NUM_RUNS nGPUs GPUidShift")
    sys.exit(1)

X = []
for i in range(num_run):
    ex = (ds_setup_file, top_setup_file, i)
    X.append(ex)
print(*X, sep='\n')

runMultiGPU(EX, X, nGPU, shift)