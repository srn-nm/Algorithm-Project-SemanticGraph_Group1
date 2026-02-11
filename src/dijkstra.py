import heapq

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
