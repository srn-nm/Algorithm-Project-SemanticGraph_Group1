import unittest
import heapq
from typing import Dict, List, Tuple, Optional

def dijkstra(graph, start, goal):
    dist = {v: float('inf') for v in graph.nodes}
    parent = {}
    dist[start] = 0

    pq = [(0, start)]
    while pq:
        d, v = heapq.heappop(pq)
        if v == goal:
            path = []
            while v in parent:
                path.append(v)
                v = parent[v]
            path.append(start)
            return path[::-1], dist[goal]

        if d > dist[v]:
            continue

        for u, w in graph.neighbors(v):
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                parent[u] = v
                heapq.heappush(pq, (dist[u], u))

    return None, float('inf')

class WeightedGraph:
    def __init__(self):
        self.nodes = ['A', 'B', 'C', 'D', 'E', 'G']  # G = goal
        self.edges = {
            'A': [('B', 2), ('C', 5)],
            'B': [('A', 2), ('D', 4), ('E', 2)],
            'C': [('A', 5), ('E', 1)],
            'D': [('B', 4), ('G', 3)],
            'E': [('B', 2), ('C', 1), ('G', 6)],
            'G': []
        }

    def neighbors(self, node):
        return self.edges.get(node, [])

class TestDijkstra(unittest.TestCase):

    def setUp(self):
        self.graph = WeightedGraph()

    def test_dijkstra_finds_shortest_path(self):
        path, cost = dijkstra(self.graph, 'A', 'G')
        self.assertIsNotNone(path)
        self.assertEqual(path, ['A', 'B', 'D', 'G'])
        self.assertEqual(cost, 9)

    def test_dijkstra_start_is_goal(self):
        path, cost = dijkstra(self.graph, 'C', 'C')
        self.assertEqual(path, ['C'])
        self.assertEqual(cost, 0)

    def test_dijkstra_no_path(self):
        self.graph.edges['D'] = [('B', 4)]
        self.graph.edges['E'] = [('B', 2), ('C', 1)]
        path, cost = dijkstra(self.graph, 'A', 'G')
        self.assertIsNone(path)
        self.assertEqual(cost, float('inf'))

    def test_dijkstra_multiple_paths(self):
        self.graph.edges['A'].append(('G', 15))
        path, cost = dijkstra(self.graph, 'A', 'G')
        self.assertEqual(path, ['A', 'B', 'D', 'G'])
        self.assertEqual(cost, 9)

    def test_dijkstra_with_invalid_start(self):
        path, cost = dijkstra(self.graph, 'X', 'G')
        self.assertIsNone(path)
        self.assertEqual(cost, float('inf'))

    def test_dijkstra_with_invalid_goal(self):
        path, cost = dijkstra(self.graph, 'A', 'X')
        self.assertIsNone(path)
        self.assertEqual(cost, float('inf'))


if __name__ == '__main__':
    unittest.main()