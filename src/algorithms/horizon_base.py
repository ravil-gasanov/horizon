from abc import ABC, abstractmethod
import itertools
import math

import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

from structs import DAG
from structs import PQ

class HorizonBase(ABC):
    def __init__(self):
        self.latent_prefix = 'z'
        self.latent_counter = 0
        self.latent_label = self.latent_prefix + str(self.latent_counter)

        self.MIN_CONFSET_SIZE = 2
        
        self.dags_scored = 0

    def discover(self, data, path):
        self._init(data, path)

        self._init_edge_scoring()
        self._forward_search()
        self._backward_search()
        print(f"DAGs scored: {self.dags_scored}")

        return self.dag
    
    def _preprocess(self, data):
        scaler = StandardScaler()

        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    def _log(self, msg):
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def _init(self, data, path):
        self.log_path = path + "/slow_horizon_log.txt"

        self._log(f"Causal model: {self.model}")
        self._log(f"Confounded model: {self.z_model}")

        self.data = self._preprocess(data)

        self.N, self.M = self.data.shape
        
        self.dag = DAG()
        self.pq = PQ()

        self.valid_phi = {}

        self.ALPHA = 0.01

        self.dag.add_nodes_from(self.data.columns)

        self.base_score = self._score_dag(self.dag)
    
    def _init_edge_scoring(self):
        nodes = list(self.data.columns)

        for i in range(self.M - 1):
            for j in range(i+1, self.M):
                a, b = nodes[i], nodes[j]
                self._update_edge_score(a, b)
        
        self._log("### Initial Candidates ###")
        for phi, delta, source, target in list(self.pq.queue):
            self._log(f"{source} --> {target} : phi = {phi}, delta = {delta}")
        
        self._log(f"Base score: {self.base_score}")
    
    def search_phi(self, conf_set, conf_delta):
        delta_phi_table = {}
        conf_phi = 1e9

        for phi, delta, s, t in list(self.pq.queue):
            delta_phi_table[(s, t)] = (delta, phi)

        for source in list(self.data.columns):
            for target in conf_set:
                if source != target:
                    try:
                        next_dag = self.dag.copy().add_edge(source, target)
                    except:
                        continue
                    
                    if self.valid_phi[(source, target)] is not None:
                        delta, phi = delta_phi_table[(source, target)]
                    else:
                        delta = self._delta(next_dag)
                    
                    conf_phi = min(conf_delta - delta, conf_phi)
        
        return conf_phi



    def _forward_search(self):
        self._log("### Forward Search ###")

        while not self.pq.empty():
            self._log(f"Base score = {self.base_score}")

            phi, delta, source, target = self.pq.pop()
            print(f"Number of candidates in pq : {self.pq.qsize()}")
            print(f"Base score: {self.base_score}")
            print(f"{source} --> {target}")
            print(f"Phi: {phi}")
            print(f"Delta: {delta}")

            if not (self._edge_is_valid(source, target) and self._phi_is_valid(phi, source, target)):
                continue

            next_dag = self.dag.copy().add_edge(source, target)

            conf_delta, conf_dag, conf_set = self._conf_search(source, target)

            if conf_delta is None:
                self.dag = next_dag.copy()
                self._log(f"No conf set is found")
                self._log(f"Added {source} --> {target}: phi = {phi}, delta = {delta}")
                self._regular_updates(target)
                continue


            if target not in conf_set and conf_delta:
                conf_phi = self.search_phi(conf_set, conf_delta)
                print(f"side conf_phi: {conf_phi}")
            else:
                conf_phi = self._phi(conf_delta, delta)
                print(f"conf_phi: {conf_phi}")

            if (conf_phi < 0):
                if (-conf_phi < phi) and target in conf_set:
                    # we are less sure in source -> target now
                    # so we update phi and put in back into pq
                    self.pq.push([-conf_phi, delta, source, target])
                    self.valid_phi[(source, target)] = -conf_phi

                    self._log(f"{source} --> {target} back to PQ with updated phi = {-conf_phi}")
                else:
                    # conf candidate is worse than the anti-causal
                    # so we can go ahead with source -> target
                    self.dag = next_dag.copy()
                    self._log(f"Added {source} --> {target}: phi = {phi}, delta = {delta}")
                    self._regular_updates(target)
            else:
                # conf_dag has the best score
                if target not in conf_set:
                    self.pq.push([phi, delta, source, target])

                self.dag = conf_dag.copy()
                self._log(f"Added confounder {self.latent_label} --> {conf_set}: phi = {conf_phi}, delta = {conf_delta}")
                self._conf_updates(conf_set)
        
        self._log(f"Base score: {self.base_score}")

    def _backward_search(self):
        self._log("### Backward Search ###")

        next_best_score = self.base_score
        next_best_dag = self.dag.copy()
        pruned_parent = None

        for node in list(self.dag.nodes):
            node_updated = True

            while node_updated:
                node_updated = False

                if node not in list(self.dag.predecessors(node)) or len(list(self.dag.predecessors(node))) < 2:
                    break

                parents = list(self.dag.predecessors(node))

                for parent in parents:
                    next_dag = self.dag.copy()

                    if parent.startswith("z") and len(list(next_dag.successors(parent))) <= 2:
                        next_dag.remove_node(parent)
                    else:
                        next_dag.remove_edge(parent, node)

                    next_score = self._score_dag(next_dag)

                    if next_score < next_best_score:
                        next_best_score = next_score
                        next_best_dag = next_dag.copy()

                        pruned_parent = parent
                
                if next_best_score < self.base_score:
                    self.base_score = next_best_score
                    self.dag = next_best_dag.copy()

                    node_updated = True
                    
                    if pruned_parent not in list(self.dag.nodes):
                        print(f"Pruned latent node {pruned_parent}")
                        self._log(f"Pruned latent node {pruned_parent}")
                    else:
                        print(f"Pruned edge: {pruned_parent} -> {node}")
                        self._log(f"Pruned edge: {pruned_parent} -> {node}")
                        
                    
                    print(f"New score: {self.base_score}")
                    self._log(f"New base score: {self.base_score}")

    
    def _phi_is_valid(self, phi, a, b):
        return self.valid_phi[(a, b)] == phi

    def _phi(self, delta_1, delta_2):
        return delta_1 - delta_2

    def _test_significance(self, delta):
        return np.power(2, -delta) <= self.ALPHA

    def _delta(self, dag):
        score = self._score_dag(dag)
        delta = max(0, self.base_score - score)
        significant = self._test_significance(delta)

        if not significant:
            delta = 0
        
        return delta

    def _update_edge_score(self, a, b):
        delta_1 = self._new_edge_delta(a, b)
        delta_2 = self._new_edge_delta(b, a)

        phi = self._phi(delta_1, delta_2)

        self._add_edge_to_pq(phi, delta_1, a, b)
        self._add_edge_to_pq(-phi, delta_2, b, a)
    
    def _new_edge_delta(self, a, b):
        delta = 0

        if self._edge_is_valid(a, b):
            dag = self.dag.copy().add_edge(a, b)
            delta = self._delta(dag)

        return delta
    
    def _add_edge_to_pq(self, phi, delta, a, b):
        if delta:
            self.pq.push([phi, delta, a, b])
            self.valid_phi[(a, b)] = phi
        else:
            self.valid_phi[(a, b)] = None

    def _conf_search(self, source, target):
        self._log("--- conf_search ---")

        conf_set = self._conf_init_search(source, target)
        if len(conf_set) < 2:
            return None, None, []
        
        print(f"Initial conf_set: {conf_set}")
        self._log(f"Initial conf_set: {conf_set}")

        conf_set = self._conf_seq_search(conf_set, mode = 'forward', target = target)
        conf_set = self._conf_seq_search(conf_set, mode = 'backward', target = target)

        conf_dag = self._conf_build_graph(conf_set)
        conf_delta = self._delta(conf_dag)

        self._log("--- conf_search ended ---")

        return conf_delta, conf_dag, conf_set
    
    def _conf_seq_search(self, conf_set, mode, target):
        conf_set_updated = True

        best_dag = self._conf_build_graph(conf_set)
        best_score = self._score_dag(best_dag)

        next_best_score = best_score
        next_best_dag = best_dag.copy()
        next_best_conf_set = conf_set.copy()

        while conf_set_updated:
            conf_set_updated = False

            if mode == 'forward':
                nodes = set(self.data.columns).difference(set(conf_set))
            else:
                nodes = conf_set.copy()

            for node in nodes:
                if mode == 'backward' or self._conf_is_valid(node):
                    if mode == 'forward':
                        next_conf_set = conf_set.copy() + [node]
                    elif len(conf_set) > self.MIN_CONFSET_SIZE:
                        next_conf_set = conf_set.copy()
                        next_conf_set.remove(node)
                    else:
                        return conf_set

                    # done this way so we don't have to keep track of latent var name here
                    next_dag = self._conf_build_graph(next_conf_set) 
                    next_score = self._score_dag(next_dag)

                    if next_score < next_best_score:
                        next_best_score = next_score
                        next_best_dag = next_dag.copy()
                        next_best_conf_set = next_conf_set.copy()

                        # print(f"Best confset so far:{next_best_conf_set} with score {next_best_score}")


            if next_best_score < best_score:
                best_score = next_best_score
                best_dag = next_best_dag.copy()
                conf_set = next_best_conf_set.copy()
                print(f"Best confset so far:{conf_set} with score {best_score}")
                self._log(f"Best confset so far:{conf_set} with score {best_score}")

                conf_set_updated = True
        
        return conf_set

    def _conf_build_graph(self, conf_set):
        dag = self.dag.copy()
        dag.add_edges_from((list(zip([self.latent_label] * len(conf_set), conf_set))))

        return dag

    def _regular_updates(self, target):
        self.base_score = self._score_dag(self.dag)

        nodes = set(self.data.columns).difference(set(self.dag.predecessors(target)))

        for node in nodes:
            # update_edge_score must check for validity, because
            # e.g. target -> node <- latent is invalid for HorizonPure
            self._update_edge_score(node, target)
        
        for child in list(self.dag.successors(target)):
            reversed_child_dag = self.dag.copy()
            reversed_child_dag.remove_edge(target, child)

            try:
                reversed_child_dag.add_edge(child, target)
            except:
                continue

            delta = self._delta(reversed_child_dag)

            if delta:
                self.dag.remove_edge(target, child)
                self._update_edge_score(target, child)
                self.base_score = self._score_dag(self.dag)

                self._log(f"Remove {target} -> {child}. Update {target}, {child} scores and put back in PQ")
    
    @abstractmethod
    def _conf_init_search(self, source, target): # pragma: no cover
        raise NotImplementedError
    
    @abstractmethod
    def _score_dag(self, dag): # pragma: no cover
        raise NotImplementedError
    
    @abstractmethod
    def _edge_is_valid(self, source, target): # pragma: no cover
        '''
        Checks if source -> target edge is structurally valid
        1) does not create a cycle
        2) has not already been added
        3) etc.
        '''
        raise NotImplementedError
    
    @abstractmethod
    def _conf_is_valid(self, node): # pragma: no cover
        '''
        Checks if the node is a valid addition to the conf_set
        '''
        raise NotImplementedError
    
    @abstractmethod
    def _conf_updates(self, conf_set): # pragma: no cover
        raise NotImplementedError

