from collections import deque

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
