import numpy as np

class Graph:
    def __init__(self):
        self.graph = {}
        self.all_configs = set()

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []
            self.all_configs.add(tuple(np.frombuffer(node)))

    def add_edge(self, node1, node2, weight):
        self.add_node(node1)
        self.add_node(node2)

        if (node2,weight) not in self.graph[node1] and (node1,weight) not  in self.graph[node2]:
            self.graph[node1].append((node2, weight))
            self.graph[node2].append((node1, weight))  # For undirected graph

        # Sort neighbors based on weights
        ## THIS ADD TIMES BUT ITS OK
        self.graph[node1] = sorted(self.graph[node1], key=lambda x: x[1])
        self.graph[node2] = sorted(self.graph[node2], key=lambda x: x[1])

    def get_all_neighbors(self, node):
        return self.graph.get(node, [])
    
    def print_node_info(self):
        for node in self.graph:
            print(f"{np.frombuffer(node)} :")
            for node_c , weight in self.graph[node]:
                print(f"    [{np.frombuffer(node_c)},{weight}]")
            print("\n")

    def get_nodes(self):
        return self.all_configs
    
if __name__ == "__main__":
    graph = Graph()
    graph.add_edge('A', 'B', 5)
    graph.add_edge('B', 'C', 3)
    graph.add_edge('A', 'C', 7)

    print(graph.get_all_neighbors('A'))