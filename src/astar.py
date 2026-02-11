import heapq

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
