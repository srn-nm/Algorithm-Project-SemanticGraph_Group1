# semantic graph builder module

import datetime
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import json
import os
import logging
from dataclasses import dataclass
from .semantic_model import SemanticModel
from config import GraphType

logger = logging.getLogger(__name__)

@dataclass
class GraphMetrics:
    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    max_degree: int
    min_degree: int
    avg_weight: float
    is_connected: bool
    num_components: int
    avg_clustering: float

class GraphBuilder:
    
    def __init__(self, semantic_model: SemanticModel, config):
        self.model = semantic_model
        self.config = config
        self.graph = None
        self.adjacency_matrix = None
        self.phrases = []
        self.phrase_to_idx = {}
    
    def build_graph(self, phrases: List[str]) -> nx.Graph: #returns created semantic graph
        self.phrases = phrases
        self.phrase_to_idx = {phrase: i for i, phrase in enumerate(phrases)}
        n = len(phrases)
        
        logger.info(f"building semantic graph with {n} nodes using type: {self.config.graph_type.value}")
        
        self.adjacency_matrix = np.full((n, n), float('inf'))
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        logger.info("calculating similarity matrix...")
        similarity_matrix = self.model.compute_similarity_matrix(phrases)
        
        # creating graph based on the type chosen
        if self.config.graph_type == GraphType.FULLY_CONNECTED:
            self._build_fully_connected(similarity_matrix)
        elif self.config.graph_type == GraphType.THRESHOLD_BASED:
            self._build_threshold_based(similarity_matrix)
        elif self.config.graph_type == GraphType.TOP_K:
            self._build_top_k(similarity_matrix)
        
        #NetworkX graph
        self.graph = self._create_networkx_graph()
        
        metrics = self._calculate_metrics()
        self._log_metrics(metrics)
        
        if self.config.save_graph:
            self._save_graph()
        
        return self.graph
    
    def _build_fully_connected(self, similarity_matrix: np.ndarray):
        n = len(self.phrases)
        for i in range(n):
            for j in range(i + 1, n):
                weight = 1 - similarity_matrix[i][j]
                self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i] = weight
        
        logger.info(f"full graph with {n*(n-1)//2} edges")
    
    def _build_threshold_based(self, similarity_matrix: np.ndarray):
        n = len(self.phrases)
        edge_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= self.config.threshold:
                    weight = 1 - similarity
                    self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i] = weight
                    edge_count += 1
        
    
    def _build_top_k(self, similarity_matrix: np.ndarray):
        n = len(self.phrases)
        k = min(self.config.top_k, n - 1)
        
        for i in range(n):
            similarities = similarity_matrix[i]
            
            #removing similarity with itself
            similarities[i] = -1
            
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            for j in top_k_indices:
                weight = 1 - similarity_matrix[i][j]
                self.adjacency_matrix[i][j] = weight
            
    def _create_networkx_graph(self) -> nx.Graph:
        G = nx.Graph()
        
        for i, phrase in enumerate(self.phrases):
            G.add_node(i, label=phrase, phrase=phrase)
        
        n = len(self.phrases)
        for i in range(n):
            for j in range(i + 1, n):
                weight = self.adjacency_matrix[i][j]
                if weight != float('inf'):
                    similarity = 1 - weight
                    G.add_edge(i, j, 
                             weight=weight, 
                             similarity=similarity,
                             distance=weight)
        
        return G
    
    def _calculate_metrics(self) -> GraphMetrics:
        if self.graph is None:
            return None
        
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        density = nx.density(self.graph) if num_nodes > 1 else 0
        
        degrees = [d for _, d in self.graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        avg_weight = np.mean(weights) if weights else 0
        
        is_connected = nx.is_connected(self.graph) if num_nodes > 0 else False
        num_components = nx.number_connected_components(self.graph)
        
        try:
            avg_clustering = nx.average_clustering(self.graph)
        except:
            avg_clustering = 0
        
        return GraphMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            avg_degree=avg_degree,
            max_degree=max_degree,
            min_degree=min_degree,
            avg_weight=avg_weight,
            is_connected=is_connected,
            num_components=num_components,
            avg_clustering=avg_clustering
        )
    
    def _log_metrics(self, metrics: GraphMetrics):
        logger.info("Graph Metrics:")
        logger.info(f" number of nodes: {metrics.num_nodes}")
        logger.info(f" number of edges: {metrics.num_edges}")
        logger.info(f" density: {metrics.density:.4f}")
        logger.info(f" average degree: {metrics.avg_degree:.2f}")
        logger.info(f" maximum degree: {metrics.max_degree}")
        logger.info(f" minimum degree: {metrics.min_degree}")
        logger.info(f" average weight: {metrics.avg_weight:.4f}")
        logger.info(f" connected?: {'yes' if metrics.is_connected else 'no'}")
        logger.info(f" number of components: {metrics.num_components}")
        logger.info(f" average clustering: {metrics.avg_clustering:.4f}")
    
    def _save_graph(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"semantic_graph_{timestamp}"
        
        #saving in NetworkX format
        graph_path = os.path.join(self.config.project.results_dir, "graphs", f"{filename}.gpickle")
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        nx.write_gpickle(self.graph, graph_path)
        
        adj_path = os.path.join(self.config.project.results_dir, "graphs", f"{filename}_adjacency.npy")
        np.save(adj_path, self.adjacency_matrix)
        
        meta = {
            "phrases": self.phrases,
            "phrase_to_idx": self.phrase_to_idx,
            "config": {
                "graph_type": self.config.graph_type.value,
                "threshold": self.config.threshold,
                "top_k": self.config.top_k
            },
            "timestamp": timestamp
        }
        
        meta_path = os.path.join(self.config.project.results_dir, "graphs", f"{filename}_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        logger.info(f"graph saved in {graph_path}")
    
    def load_graph(self, path: str):
        self.graph = nx.read_gpickle(path)
        logger.info(f"loading graph from {path}")
    
    def get_neighbors(self, node_idx: int, max_neighbors: Optional[int] = None) -> List[Tuple[int, float]]: # returns a list of (adjenc, weight)
        if self.graph is None:
            return []
        
        neighbors = list(self.graph.neighbors(node_idx))
        neighbor_weights = []
        
        for neighbor in neighbors:
            weight = self.graph[node_idx][neighbor]['weight']
            neighbor_weights.append((neighbor, weight))
        
        # the least weight comes first
        neighbor_weights.sort(key=lambda x: x[1])
        
        if max_neighbors:
            neighbor_weights = neighbor_weights[:max_neighbors]
        
        return neighbor_weights
    
    def get_shortest_paths_all_pairs(self) -> Dict[Tuple[int, int], List[int]]: #returns dictionary of pathes
        if self.graph is None:
            return {}
        
        paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight='weight'))
        return paths