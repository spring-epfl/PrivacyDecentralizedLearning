from . import *
from DL_attacks.FL import FederatedAVG

nu = 36

CDL = FederatedAVG
name = f"federated{nu}"

federated = True
lrd = [None]