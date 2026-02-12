"""
semantic graph builder module
"""
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

        if hasattr(config, 'project'):
            self.results_dir = config.project.results_dir
        elif isinstance(config, dict):
            if 'project' in config:
                if hasattr(config['project'], 'results_dir'):
                    self.results_dir = config['project'].results_dir
                elif isinstance(config['project'], dict):
                    self.results_dir = config['project'].get('results_dir', './results')
                else:
                    self.results_dir = './results'
            else:
                self.results_dir = './results'
        else:
            self.results_dir = './results'
    
    def build_graph(self, phrases: List[str]) -> nx.Graph:
        """Build semantic graph from phrases"""
        self.phrases = phrases
        self.phrase_to_idx = {phrase: i for i, phrase in enumerate(phrases)}
        n = len(phrases)
        
        logger.info(f"Building semantic graph with {n} nodes using type: {self.config.graph_type.value}")
        
        self.adjacency_matrix = np.full((n, n), float('inf'))
        np.fill_diagonal(self.adjacency_matrix, 0)
        
        logger.info("Calculating similarity matrix...")
        similarity_matrix = self.model.compute_similarity_matrix(phrases)

        if self.config.graph_type == GraphType.FULLY_CONNECTED:
            self._build_fully_connected(similarity_matrix)
        elif self.config.graph_type == GraphType.THRESHOLD_BASED:
            self._build_threshold_based(similarity_matrix)
        elif self.config.graph_type == GraphType.TOP_K:
            self._build_top_k(similarity_matrix)
        else:
            logger.warning(f"Unknown graph type: {self.config.graph_type}, using threshold based")
            self._build_threshold_based(similarity_matrix)

        self.graph = self._create_networkx_graph()
        
        metrics = self._calculate_metrics()
        self._log_metrics(metrics)

        should_save = False
        if hasattr(self.config, 'save_graph'):
            should_save = self.config.save_graph
        elif isinstance(self.config, dict):
            should_save = self.config.get('save_graph', False)
        
        if should_save:
            self._save_graph()
        
        return self.graph
    
    def _build_fully_connected(self, similarity_matrix: np.ndarray):
        """Build fully connected graph"""
        n = len(self.phrases)
        for i in range(n):
            for j in range(i + 1, n):
                weight = 1 - similarity_matrix[i][j]
                self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i] = weight
        
        logger.info(f"Fully connected graph with {n*(n-1)//2} edges")
    
    def _build_threshold_based(self, similarity_matrix: np.ndarray):
        """Build graph based on similarity threshold"""
        n = len(self.phrases)
        edge_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= self.config.threshold:
                    weight = 1 - similarity
                    self.adjacency_matrix[i][j] = self.adjacency_matrix[j][i] = weight
                    edge_count += 1
        
        logger.info(f"Threshold-based graph with {edge_count} edges (threshold={self.config.threshold})")
    
    def _build_top_k(self, similarity_matrix: np.ndarray):
        """Build graph with top-k most similar neighbors"""
        n = len(self.phrases)
        k = min(self.config.top_k, n - 1)
        edge_count = 0
        
        for i in range(n):
            similarities = similarity_matrix[i].copy()

            similarities[i] = -1

            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            for j in top_k_indices:
                if similarities[j] >= 0:
                    weight = 1 - similarity_matrix[i][j]
                    self.adjacency_matrix[i][j] = weight
                    edge_count += 1
        
        logger.info(f"Top-{k} graph with {edge_count} edges")
    
    def _create_networkx_graph(self) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph"""
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
        """Calculate graph metrics"""
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
        """Log graph metrics"""
        if metrics is None:
            logger.warning("No metrics to log")
            return
        
        logger.info("=" * 50)
        logger.info("Graph Metrics:")
        logger.info("=" * 50)
        logger.info(f"  Number of nodes: {metrics.num_nodes}")
        logger.info(f"  Number of edges: {metrics.num_edges}")
        logger.info(f"  Density: {metrics.density:.4f}")
        logger.info(f"  Average degree: {metrics.avg_degree:.2f}")
        logger.info(f"  Maximum degree: {metrics.max_degree}")
        logger.info(f"  Minimum degree: {metrics.min_degree}")
        logger.info(f"  Average weight: {metrics.avg_weight:.4f}")
        logger.info(f"  Connected: {'Yes' if metrics.is_connected else 'No'}")
        logger.info(f"  Number of components: {metrics.num_components}")
        logger.info(f"  Average clustering: {metrics.avg_clustering:.4f}")
        logger.info("=" * 50)
    
    def _save_graph(self):
        """Save graph to disk - NetworkX 3.0+ compatible"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"semantic_graph_{timestamp}"

        graphs_dir = os.path.join(self.results_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)

        graphml_path = os.path.join(graphs_dir, f"{filename}.graphml")
        try:
            nx.write_graphml(self.graph, graphml_path)
            logger.info(f"Graph saved as GraphML to {graphml_path}")
        except Exception as e:
            logger.warning(f"Could not save as GraphML: {e}")

        gexf_path = os.path.join(graphs_dir, f"{filename}.gexf")
        try:
            nx.write_gexf(self.graph, gexf_path)
            logger.info(f"Graph saved as GEXF to {gexf_path}")
        except Exception as e:
            logger.warning(f"Could not save as GEXF: {e}")

        json_path = os.path.join(graphs_dir, f"{filename}.json")
        try:
            graph_data = {
                "nodes": [],
                "edges": []
            }

            for node in self.graph.nodes():
                node_data = {
                    "id": node,
                    "label": self.phrases[node] if node < len(self.phrases) else f"Node {node}",
                    "phrase": self.phrases[node] if node < len(self.phrases) else f"Node {node}"
                }
                graph_data["nodes"].append(node_data)

            for u, v, data in self.graph.edges(data=True):
                edge_data = {
                    "source": u,
                    "target": v,
                    "weight": data.get('weight', 0),
                    "similarity": data.get('similarity', 0),
                    "distance": data.get('distance', 0)
                }
                graph_data["edges"].append(edge_data)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Graph saved as JSON to {json_path}")
        except Exception as e:
            logger.warning(f"Could not save as JSON: {e}")
        adj_path = os.path.join(graphs_dir, f"{filename}_adjacency.npy")
        np.save(adj_path, self.adjacency_matrix)
        logger.info(f"Adjacency matrix saved to {adj_path}")

        meta = {
            "phrases": self.phrases,
            "phrase_to_idx": self.phrase_to_idx,
            "config": {
                "graph_type": self.config.graph_type.value if hasattr(self.config.graph_type, 'value') else str(self.config.graph_type),
                "threshold": getattr(self.config, 'threshold', 0.25),
                "top_k": getattr(self.config, 'top_k', 8)
            },
            "timestamp": timestamp,
            "metrics": {
                "num_nodes": len(self.phrases),
                "num_edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph)
            },
            "saved_formats": ["adjacency", "json"]  # Add more formats as they succeed
        }
        
        meta_path = os.path.join(graphs_dir, f"{filename}_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metadata saved to {meta_path}")
    
    def load_graph(self, path: str) -> Optional[nx.Graph]:
        """Load graph from disk - supports multiple formats"""

        loaders = [
            ('.graphml', nx.read_graphml),
            ('.gexf', nx.read_gexf),
            ('.gpickle', nx.read_gpickle),  # For older NetworkX versions
            ('.json', self._load_graph_from_json)
        ]
        
        for ext, loader in loaders:
            if path.endswith(ext):
                try:
                    self.graph = loader(path)
                    logger.info(f"Graph loaded from {path}")
                    return self.graph
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        for ext, loader in loaders:
            try_path = path + ext if not path.endswith(ext) else path
            if os.path.exists(try_path):
                try:
                    self.graph = loader(try_path)
                    logger.info(f"Graph loaded from {try_path}")
                    return self.graph
                except:
                    continue
        
        logger.error(f"Could not load graph from {path}")
        return None
    
    def _load_graph_from_json(self, path: str) -> nx.Graph:
        """Load graph from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        G = nx.Graph()

        for node_data in graph_data.get('nodes', []):
            node_id = node_data['id']
            G.add_node(node_id, 
                      label=node_data.get('label', f"Node {node_id}"),
                      phrase=node_data.get('phrase', f"Node {node_id}"))

        for edge_data in graph_data.get('edges', []):
            G.add_edge(edge_data['source'], 
                      edge_data['target'],
                      weight=edge_data.get('weight', 0),
                      similarity=edge_data.get('similarity', 0),
                      distance=edge_data.get('distance', 0))
        
        return G
    
    def get_neighbors(self, node_idx: int, max_neighbors: Optional[int] = None) -> List[Tuple[int, float]]:
        """Get neighbors of a node with their weights"""
        if self.graph is None:
            logger.warning("Graph not built yet")
            return []
        
        if node_idx not in self.graph:
            logger.warning(f"Node {node_idx} not in graph")
            return []
        
        neighbors = list(self.graph.neighbors(node_idx))
        neighbor_weights = []
        
        for neighbor in neighbors:
            weight = self.graph[node_idx][neighbor]['weight']
            neighbor_weights.append((neighbor, weight))

        neighbor_weights.sort(key=lambda x: x[1])
        
        if max_neighbors and max_neighbors > 0:
            neighbor_weights = neighbor_weights[:max_neighbors]
        
        return neighbor_weights
    
    def get_shortest_paths_all_pairs(self) -> Dict[Tuple[int, int], List[int]]:
        """Compute all-pairs shortest paths"""
        if self.graph is None:
            logger.warning("Graph not built yet")
            return {}
        
        try:
            paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight='weight'))
            return paths
        except Exception as e:
            logger.error(f"Error computing all-pairs shortest paths: {e}")
            return {}
    
    def get_graph_summary(self) -> Dict:
        """Get graph summary as dictionary"""
        if self.graph is None:
            return {"error": "Graph not built"}
        
        metrics = self._calculate_metrics()
        if metrics is None:
            return {"error": "Could not calculate metrics"}
        
        summary = {
            "num_nodes": metrics.num_nodes,
            "num_edges": metrics.num_edges,
            "density": metrics.density,
            "avg_degree": metrics.avg_degree,
            "max_degree": metrics.max_degree,
            "min_degree": metrics.min_degree,
            "avg_weight": metrics.avg_weight,
            "is_connected": metrics.is_connected,
            "num_components": metrics.num_components,
            "avg_clustering": metrics.avg_clustering,
            "graph_type": self.config.graph_type.value if hasattr(self.config.graph_type, 'value') else str(self.config.graph_type),
            "threshold": getattr(self.config, 'threshold', 0.25),
            "top_k": getattr(self.config, 'top_k', 8)
        }
        
        return summary