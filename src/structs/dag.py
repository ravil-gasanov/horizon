import networkx as nx

class DAG(nx.DiGraph):
    def add_edge(self, a, b):
        if self.has_edge(a, b):
            raise ValueError("Already has this edge")

        super().add_edge(a, b)

        if not nx.is_directed_acyclic_graph(self):
            self.remove_edge(a, b)
            raise ValueError("Adding the edge creates a cycle")
        
        return self