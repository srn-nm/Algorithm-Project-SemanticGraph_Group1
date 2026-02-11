from src.graph_builder import Graph
from src.bfs import bfs
from src.dijkstra import dijkstra
from src.astar import a_star

g = Graph()
nodes = ["Fruit", "Apple", "Technology", "iPhone"]
for n in nodes:
    g.add_node(n)

g.add_edge("Fruit", "Apple", 0.2)
g.add_edge("Apple", "Technology", 0.3)
g.add_edge("Fruit", "Technology", 0.9)
g.add_edge("Technology", "iPhone", 0.1)

# BFS
path_bfs = bfs(g, "Fruit", "iPhone")
print("BFS path:", path_bfs)

# Dijkstra
path_dij, cost_dij = dijkstra(g, "Fruit", "iPhone")
print("Dijkstra path:", path_dij, "cost:", cost_dij)

# A*
def heuristic(v, goal):
    h_vals = {"Fruit": 0.6, "Apple": 0.4, "Technology": 0.1, "iPhone": 0}
    return h_vals[v]

path_astar, cost_astar = a_star(g, "Fruit", "iPhone", heuristic)
print("A* path:", path_astar, "cost:", cost_astar)
