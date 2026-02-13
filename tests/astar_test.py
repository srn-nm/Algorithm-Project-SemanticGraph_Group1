import unittest
import heapq
from typing import Dict, List, Tuple

def a_star(graph, start, goal, heuristic):
    g = {v: float('inf') for v in graph.nodes}
    f = {v: float('inf') for v in graph.nodes}
    parent = {}

    g[start] = 0
    f[start] = heuristic(start, goal)

    open_set = [(f[start], start)]

    while open_set:
        _, v = heapq.heappop(open_set)
        if v == goal:
            path = []
            while v in parent:
                path.append(v)
                v = parent[v]
            path.append(start)
            return path[::-1], g[goal]

        for u, w in graph.neighbors(v):
            tentative_g = g[v] + w
            if tentative_g < g[u]:
                g[u] = tentative_g
                f[u] = g[u] + heuristic(u, goal)
                parent[u] = v
                heapq.heappush(open_set, (f[u], u))

    return None, float('inf')

class SimpleGraph:
    def __init__(self):
        self.nodes = ['A', 'B', 'C', 'D', 'E', 'G']  # G = goal
        self.edges = {
            'A': [('B', 2), ('C', 5)],
            'B': [('D', 4), ('E', 2)],
            'C': [('E', 1)],
            'D': [('G', 3)],
            'E': [('G', 6)],
            'G': []
        }

    def neighbors(self, node):
        return self.edges.get(node, [])

def heuristic(node, goal):
    h_vals = {
        'A': 5,
        'B': 3,
        'C': 4,
        'D': 2,
        'E': 3,
        'G': 0
    }
    return h_vals.get(node, 0)

class TestAStar(unittest.TestCase):

    def setUp(self):
        self.graph = SimpleGraph()
        self.start = 'A'
        self.goal = 'G'

    def test_a_star_finds_path(self):
        path, cost = a_star(self.graph, self.start, self.goal, heuristic)
        self.assertIsNotNone(path)
        self.assertEqual(path, ['A', 'B', 'D', 'G'])
        self.assertEqual(cost, 2 + 4 + 3)  # A->B (2), B->D (4), D->G (3) = 9

    def test_a_star_no_path(self):
        graph_no_path = SimpleGraph()
        graph_no_path.edges['D'] = []
        graph_no_path.edges['E'] = []
        path, cost = a_star(graph_no_path, 'A', 'G', heuristic)
        self.assertIsNone(path)
        self.assertEqual(cost, float('inf'))

    def test_a_star_start_is_goal(self):
        path, cost = a_star(self.graph, 'G', 'G', heuristic)
        self.assertEqual(path, ['G'])
        self.assertEqual(cost, 0)

    def test_heuristic_called(self):
        call_count = {'count': 0}
        def heuristic_with_count(node, goal):
            call_count['count'] += 1
            return heuristic(node, goal)
        a_star(self.graph, 'A', 'G', heuristic_with_count)
        self.assertGreater(call_count['count'], 0)


if __name__ == '__main__':
    unittest.main()