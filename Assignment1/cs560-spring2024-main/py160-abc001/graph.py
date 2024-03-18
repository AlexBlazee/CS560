class Graph:
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, node1, node2, weight):
        self.add_node(node1)
        self.add_node(node2)
        self.graph[node1].append((node2, weight))
        self.graph[node2].append((node1, weight))  # For undirected graph

        # Sort neighbors based on weights
        self.graph[node1] = sorted(self.graph[node1], key=lambda x: x[1])
        self.graph[node2] = sorted(self.graph[node2], key=lambda x: x[1])

    def get_all_neighbors(self, node):
        return self.graph.get(node, [])
    
if __name__ == "__main__":
    graph = Graph()
    graph.add_edge('A', 'B', 5)
    graph.add_edge('B', 'C', 3)
    graph.add_edge('A', 'C', 7)

    print(graph.get_neighbors('A'))