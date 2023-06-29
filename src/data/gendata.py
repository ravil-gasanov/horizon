import random

import numpy as np
import networkx as nx
from cdt.data import AcyclicGraphGenerator


class DataDagGenerator:
    def __init__(self):
        self.MEAN = 0
        self.STD = 1
    
    def generate(self, path, n, m, cs_sizes, confounder_strength, cm = "linear", zm = "linear"):
        self.path = path
        self.n = n
        self.m = m
        self.cs_sizes = cs_sizes
        self.confounder_strength = confounder_strength
        self.cm = cm
        self.zm = zm

        self.latent_index = 0

        for cs_size in self.cs_sizes:
            assert cs_size <= m

        self.generator = AcyclicGraphGenerator(causal_mechanism = cm, noise='gaussian', noise_coeff = 0.01,\
        npoints=n, nodes=m, parents_max=3, expected_degree=1, dag_type='default')
    
        self.data, self.dag = self.generator.generate()

        for cs_size in self.cs_sizes:
            self.confound(cs_size)

        self.log()
        
        return self.data, self.dag
    
    def confound(self, cs_size):
        nodes = list(self.dag.nodes)
        cs_nodes = random.sample(nodes, cs_size)

        latent_label = f"z{self.latent_index}"

        self.dag.add_edges_from((list(zip([latent_label] * cs_size, cs_nodes))))

        self.z = np.random.normal(self.MEAN, self.STD, (self.n, 1))
        self.bias = np.random.normal(self.MEAN, self.STD, (1, cs_size))
        self.w = np.random.normal(self.MEAN, self.STD, (1, cs_size))

        self.data[latent_label] = self.z

        if self.zm == "linear":
            self.data[cs_nodes] += self.bias + self.confounder_strength * self.z * self.w
        else:
            kernels = self.generate_kernels()

            for j, node in enumerate(cs_nodes):
                self.data[[node]] += self.bias[:, j] + self.confounder_strength * self.w[:, j] * kernels[j](self.z)
        
        self.latent_index += 1

    def generate_kernels(self):
        poly_kernels = [lambda x: np.power(x, 1),\
                        lambda x: np.power(x, 2),\
                        lambda x: np.power(x, 3),\
                        lambda x: np.power(x, 4),\
                        lambda x: np.power(x, 5),\
                        ]
        
        all_kernels = poly_kernels + [np.exp, np.sin, np.cos]

        if self.zm == "polynomial":
            return np.random.choice(poly_kernels, cs_size)
        else:
            return np.random.choice(all_kernels, cs_size)
    
    def log(self):
        self.data.to_csv(self.path + "/data.csv", index = False)
        nx.write_adjlist(self.dag, self.path + "/true_dag.adjlist")

        # To be continued...
        # log weights, biases, kernels
        # log self.generator