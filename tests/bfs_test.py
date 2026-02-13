import unittest
from collections import deque
from typing import List, Tuple

# ---------- تابع BFS ----------
def bfs(graph, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        v = queue.popleft()
        if v == goal:
            path = []
            while v is not None:
                path.append(v)
                v = parent[v]
            return path[::-1]
        for u, _ in graph.neighbors(v):
            if u not in visited:
                visited.add(u)
                parent[u] = v
                queue.append(u)
    return None

# ---------- گراف ساده ----------
class SimpleGraph:
    def __init__(self):
        self.nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.edges = {
            'A': [('B', 2), ('C', 5)],
            'B': [('A', 2), ('D', 4), ('E', 1)],
            'C': [('A', 5), ('E', 3)],
            'D': [('B', 4), ('F', 2), ('G', 3)],
            'E': [('B', 1), ('C', 3), ('G', 6)],
            'F': [('D', 2)],
            'G': [('D', 3), ('E', 6)]
        }

    def neighbors(self, node):
        return self.edges.get(node, [])

# ---------- کلاس تست ----------
class TestBFS(unittest.TestCase):

    def setUp(self):
        self.graph = SimpleGraph()

    def test_bfs_finds_path(self):
        path = bfs(self.graph, 'A', 'G')
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'G')

    def test_bfs_shortest_path_unweighted(self):
        path = bfs(self.graph, 'A', 'G')
        self.assertEqual(len(path) - 1, 3)

    def test_bfs_no_path(self):
        self.graph.edges['D'] = [('B', 4), ('F', 2)]
        self.graph.edges['E'] = [('B', 1), ('C', 3)]
        path = bfs(self.graph, 'A', 'G')
        self.assertIsNone(path)

    def test_bfs_start_is_goal(self):
        path = bfs(self.graph, 'C', 'C')
        self.assertEqual(path, ['C'])

    def test_bfs_visited_prevents_cycles(self):
        path = bfs(self.graph, 'A', 'F')
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'F')
        self.assertLessEqual(len(path), 5)

    def test_bfs_with_invalid_start(self):
        path = bfs(self.graph, 'X', 'G')
        self.assertIsNone(path)

    def test_bfs_with_invalid_goal(self):
        path = bfs(self.graph, 'A', 'X')
        self.assertIsNone(path)

if __name__ == '__main__':
    unittest.main()