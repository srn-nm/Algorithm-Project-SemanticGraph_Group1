import heapq
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import time
import logging
from .semantic_model import SemanticModel
from config import Algorithm

logger = logging.getLogger(__name__)

class SearchResult:
    
    def __init__(self, success: bool = False):
        self.success = success
        self.path = [] 
        self.path_phrases = []  
        self.total_distance = float('inf')
        self.nodes_visited = 0
        self.execution_time = 0.0
        self.algorithm = ""
        self.error_message = ""
        
        self.edge_details = []
    
    def __repr__(self):
        if not self.success:
            return f"SearchResult(success=False, error='{self.error_message}')"
        
        path_str = " → ".join(self.path_phrases[:3])
        if len(self.path_phrases) > 3:
            path_str += f" → ... → {self.path_phrases[-1]}"
        
        return (f"SearchResult(algorithm={self.algorithm}, "
                f"distance={self.total_distance:.4f}, "
                f"nodes={self.nodes_visited}, "
                f"time={self.execution_time:.4f}s, "
                f"path={path_str})")

class SearchAlgorithms:
    
    def __init__(self, graph_builder, config, semantic_model: Optional[SemanticModel] = None):
       
        self.graph_builder = graph_builder
        self.config = config
        self.semantic_model = semantic_model
        self.graph = None
        self.adjacency = None
        self.phrases = []
    
    def _refresh_graph_data(self):
        """Refresh graph data from graph_builder"""
        self.graph = self.graph_builder.graph
        self.adjacency = self.graph_builder.adjacency_matrix
        self.phrases = self.graph_builder.phrases
    
    def find_path(self, start_phrase: str, end_phrase: str, algorithm: Algorithm = None) -> SearchResult:

        self._refresh_graph_data()
        
        if algorithm is None:
            algorithm = self.config.default_algorithm

        if self.graph is None or self.adjacency is None:
            result = SearchResult(success=False)
            result.error_message = "Graph not built yet"
            return result

        if not hasattr(self.graph_builder, 'phrase_to_idx') or not self.graph_builder.phrase_to_idx:
            result = SearchResult(success=False)
            result.error_message = "Phrase index not available"
            return result
        
        if start_phrase not in self.graph_builder.phrase_to_idx:
            result = SearchResult(success=False)
            result.error_message = f"Start phrase '{start_phrase}' does not exist."
            return result
        
        if end_phrase not in self.graph_builder.phrase_to_idx:
            result = SearchResult(success=False)
            result.error_message = f"End phrase '{end_phrase}' does not exist."
            return result
        
        start_idx = self.graph_builder.phrase_to_idx[start_phrase]
        end_idx = self.graph_builder.phrase_to_idx[end_phrase]

        n = len(self.phrases)
        if start_idx >= n or start_idx < 0:
            result = SearchResult(success=False)
            result.error_message = f"Start index {start_idx} out of range (0-{n-1})"
            return result
        
        if end_idx >= n or end_idx < 0:
            result = SearchResult(success=False)
            result.error_message = f"End index {end_idx} out of range (0-{n-1})"
            return result
        
        logger.info(f"Searching for path from '{start_phrase}' (idx={start_idx}) to '{end_phrase}' (idx={end_idx})")
        logger.info(f"Algorithm used: {algorithm.value}")
        
        start_time = time.time()
        
        if algorithm == Algorithm.BFS:
            result = self._bfs_search(start_idx, end_idx)
        elif algorithm == Algorithm.DIJKSTRA:
            result = self._dijkstra_search(start_idx, end_idx)
        elif algorithm == Algorithm.ASTAR:
            result = self._astar_search(start_idx, end_idx)
        else:
            result = SearchResult(success=False)
            result.error_message = f"Algorithm {algorithm} is not available"
        
        result.execution_time = time.time() - start_time
        result.algorithm = algorithm.value
        
        if result.success:
            result.path_phrases = [self.phrases[i] for i in result.path if i < len(self.phrases)]
            result.edge_details = self._get_edge_details(result.path)
        
        logger.info(f"Execution time: {result.execution_time:.4f} seconds")
        if result.success:
            logger.info(f"Total distance: {result.total_distance:.4f}")
            logger.info(f"Visited nodes: {result.nodes_visited}")
        
        return result
    
    def _bfs_search(self, start: int, end: int) -> SearchResult:
        result = SearchResult()
        
        n = len(self.phrases)
        visited = [False] * n
        parent = [-1] * n
        queue = [start]
        visited[start] = True
        nodes_visited = 0
        
        while queue:
            current = queue.pop(0)
            nodes_visited += 1
            
            if current == end:
                path = self._reconstruct_path(parent, start, end)
                result.success = True
                result.path = path
                result.total_distance = len(path) - 1 
                result.nodes_visited = nodes_visited
                return result
            
            for neighbor in range(n):
                if not visited[neighbor] and self.adjacency[current][neighbor] != float('inf'):
                    visited[neighbor] = True
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        result.success = False
        result.error_message = "No path found"
        result.nodes_visited = nodes_visited
        return result
    
    def _dijkstra_search(self, start: int, end: int) -> SearchResult:
        result = SearchResult()
        
        n = len(self.phrases)

        if start >= n or end >= n:
            result.success = False
            result.error_message = f"Start ({start}) or end ({end}) index out of range (max: {n-1})"
            return result
        
        dist = [float('inf')] * n
        dist[start] = 0
        parent = [-1] * n
        visited = [False] * n
        
        pq = [(0, start)]
        nodes_visited = 0
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            nodes_visited += 1
            
            if visited[current]:
                continue
            visited[current] = True
            
            if current == end:
                path = self._reconstruct_path(parent, start, end)
                result.success = True
                result.path = path
                result.total_distance = dist[end]
                result.nodes_visited = nodes_visited
                return result
            
            for neighbor in range(n):
                weight = self.adjacency[current][neighbor]
                if weight != float('inf'):
                    new_dist = current_dist + weight
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        parent[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
        
        result.success = False
        result.error_message = "No path found"
        result.nodes_visited = nodes_visited
        return result

    def _astar_search(self, start: int, end: int) -> SearchResult:
        result = SearchResult()

        if self.semantic_model is None:
            result.success = False
            result.error_message = "No semantic model chosen for A* heuristic"
            return result

        n = len(self.phrases)

        if start >= n or end >= n:
            result.success = False
            result.error_message = f"Start ({start}) or end ({end}) index out of range (max: {n - 1})"
            return result

        heuristic_values = [0.0] * n
        for i in range(n):
            if i == end:
                heuristic_values[i] = 0.0
            else:
                similarity = self.semantic_model.compute_similarity(
                    self.phrases[i],
                    self.phrases[end]
                ).similarity
                heuristic_values[i] = self.config.heuristic_weight * (1 - similarity)

        g_score = [float('inf')] * n
        g_score[start] = 0

        f_score = [float('inf')] * n
        f_score[start] = heuristic_values[start]  # f(start) = h(start)

        open_set = [(f_score[start], start)]
        heapq.heapify(open_set)

        came_from = [-1] * n
        nodes_visited = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_visited += 1

            if current == end:
                path = self._reconstruct_path(came_from, start, end)
                result.success = True
                result.path = path
                result.total_distance = g_score[end]
                result.nodes_visited = nodes_visited
                return result

            for neighbor in range(n):
                weight = self.adjacency[current][neighbor]
                if weight != float('inf'):
                    tentative_g = g_score[current] + weight
                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic_values[neighbor]  # استفاده از مقدار پیش‌محاسبه شده
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        result.success = False
        result.error_message = "No path found"
        result.nodes_visited = nodes_visited
        return result
    
    def _reconstruct_path(self, parent: List[int], start: int, end: int) -> List[int]:
        path = []
        current = end
        
        while current != -1:
            path.append(current)
            current = parent[current]
        
        path.reverse()

        if not path or path[0] != start:
            return []
        
        return path
    
    def _get_edge_details(self, path: List[int]) -> List[Dict]:
        details = []
        
        if len(path) < 2:
            return details
        
        for i in range(len(path) - 1):
            from_idx = path[i]
            to_idx = path[i + 1]
            
            weight = self.adjacency[from_idx][to_idx]
            similarity = 1 - weight
            
            details.append({
                'from': self.phrases[from_idx] if from_idx < len(self.phrases) else f"Node {from_idx}",
                'to': self.phrases[to_idx] if to_idx < len(self.phrases) else f"Node {to_idx}",
                'from_idx': from_idx,
                'to_idx': to_idx,
                'weight': weight,
                'similarity': similarity,
                'distance': weight
            })
        
        return details
    
    def compare_algorithms(self, start_phrase: str, end_phrase: str) -> Dict[str, SearchResult]:
        results = {}
        
        for algorithm in Algorithm:
            try:
                result = self.find_path(start_phrase, end_phrase, algorithm)
                results[algorithm.value] = result
            except Exception as e:
                result = SearchResult(success=False)
                result.error_message = str(e)
                results[algorithm.value] = result
        
        return results
    
    def find_k_shortest_paths(self, start: int, end: int, k: int = 3) -> List[SearchResult]:
        """Returns list of k shortest paths"""
        try:
            import networkx as nx
            
            # Refresh graph data
            self._refresh_graph_data()
            
            if self.graph is None:
                logger.error("Graph not built")
                return []
            
            paths = list(nx.shortest_simple_paths(
                self.graph, 
                start, 
                end, 
                weight='weight'
            ))
            
            results = []
            for i, path in enumerate(paths):
                if i >= k:
                    break
                
                result = SearchResult(success=True)
                result.path = path
                result.path_phrases = [self.phrases[idx] for idx in path if idx < len(self.phrases)]
                result.nodes_visited = len(path)
                
                total_distance = 0
                for j in range(len(path) - 1):
                    total_distance += self.adjacency[path[j]][path[j+1]]
                
                result.total_distance = total_distance
                result.edge_details = self._get_edge_details(path)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Could not find k shortest paths: {e}")
            return []