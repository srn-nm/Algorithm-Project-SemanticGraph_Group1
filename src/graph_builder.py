class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}  

    def add_node(self, node):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = []

    def add_edge(self, u, v, weight):
        self.edges[u].append((v, weight))
        self.edges[v].append((u, weight))  

    def neighbors(self, node):
        return self.edges.get(node, [])