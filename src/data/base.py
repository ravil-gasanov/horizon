import pandas as pd
import numpy as np
import networkx as nx
from structs import DAG

MEAN = 0
STD = 1

def generate_independent(n, x_size):
    X = np.random.normal(MEAN, STD, (n, x_size))
    data = pd.DataFrame(X, columns = [f"iv{i}" for i in range(x_size)])

    true_dag = DAG()
    true_dag.add_nodes_from(data.columns)

    return data, true_dag
    
