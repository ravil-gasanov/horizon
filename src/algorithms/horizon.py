import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from scipy.stats import pearsonr


from algorithms import HorizonBase
from mdl import score_residuals
from mdl import MarsMDL, ZMarsMDL


class Horizon(HorizonBase):
    def __init__(self, model = MarsMDL(), z_model = PCA(n_components = 1)):
        super().__init__()

        self.model = model
        self.z_model = z_model
        
    def _edge_is_valid(self, source, target):
        dag = self.dag.copy()

        try:
            dag.add_edge(source, target)
        except:
            return False
        
        return True
    
    def _correlates_with_both(self, x, source, target):
        # TO DO: use HSIC
        x1, x2, y = self.data[[source]], self.data[[target]], self.data[[x]]
        x1, x2, y = np.squeeze(x1), np.squeeze(x2), np.squeeze(y)
        _, pval1 = pearsonr(x1, y)
        _, pval2 = pearsonr(x2, y)

        return (pval1 < self.ALPHA) and (pval2 < self.ALPHA)

    def _conf_init_search(self, source, target):
        conf_set = []

        if self._conf_is_valid(target):
            conf_set.append(target)
        else:
            return conf_set
        
        nodes = list(self.data.columns)
        nodes.remove(target)

        for node in nodes:
            if self._conf_is_valid(node) and self._correlates_with_both(node, source, target):
                conf_set.append(node)
        
        return conf_set

    def _conf_is_valid(self, node):
        return not node.startswith("z")

    def _conf_updates(self, conf_set):
        self.latent_counter += 1
        self.latent_label = self.latent_prefix + str(self.latent_counter)
        
        self.base_score = self._score_dag(self.dag.copy())
            
        for target in conf_set:
            self._regular_updates(target)
    
    def _generate_z(self, dag):
        Z = pd.DataFrame()

        for j in range(self.latent_counter + 1):
            latent_node = self.latent_prefix + str(j)

            if latent_node not in list(dag.nodes):
                continue

            conf_set = list(dag.successors(latent_node))
            
            CS = self.data[conf_set].values

            z = self.z_model.fit_transform(CS)
            z = z.reshape(-1, 1)
            
            Z[[latent_node]] = z
        
        return Z

    def _score_dag(self, dag):
        self.dags_scored += 1
        score = 0

        Z = self._generate_z(dag)

        for latent_node in Z.columns:
            z = Z[[latent_node]].values
            score += score_residuals(z)

    
        nodes = self.data.columns
        

        for target in nodes:
            parents = list(dag.predecessors(target))

            if len(parents) == 0:
                score += self._score_independent(target)
            else:
                X = pd.DataFrame()
                for parent in parents:
                    if parent[0] == 'z':
                        try:
                            X[[parent]] = Z[[parent]]
                        except KeyError:
                            print("!!!!!!!!!!!!!!!! something wrong !!!!!!!!!!!!!!!")
                            print(parents)
                            print(dag.nodes)
                            print(dag.edges)
                            self._log(f"!!!!!!!!!!!!! {parent} index error !!!!!!!!!!!")

                    else:
                        X[[parent]] = self.data[[parent]]
                
                X = X.values
                y = self.data[[target]].values

                score += self.model.score(X, y, self.M + self.latent_counter)
        
        return score

    def _score_independent(self, target):
        x = self.data[[target]].values
        
        return score_residuals(x)
