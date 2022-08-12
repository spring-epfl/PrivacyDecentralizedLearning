from . import *
from DL_attacks.FL import FederatedAVG
import networkx as nx

G = nx.davis_southern_women_graph()
nu = len(G)

CDL = FederatedAVG
name = f"federated_davis"

federated = True
lrd = [None]