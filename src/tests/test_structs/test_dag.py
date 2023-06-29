from structs import DAG

class TestDAG:
    def test_detect_cycle(self):
        dag = DAG()
        dag.add_edge('x1', 'x2')
        cycle_detected = False

        try:
            dag.add_edge('x2', 'x1')
        except ValueError:
            cycle_detected = True
        
        assert cycle_detected
    
    def test_detect_duplicate_edge(self):
        dag = DAG()
        dag.add_edge('x1', 'x2')
        duplicate_edge_detected = False

        try:
            dag.add_edge('x1', 'x2')
        except ValueError:
            duplicate_edge_detected = True
        
        assert duplicate_edge_detected