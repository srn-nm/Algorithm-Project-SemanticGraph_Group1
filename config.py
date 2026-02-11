import os
from dataclasses import dataclass
from enum import Enum

class GraphType(Enum):
    FULLY_CONNECTED = "fully_connected"
    THRESHOLD_BASED = "threshold_based"
    TOP_K = "top_k"

class Algorithm(Enum):
    BFS = "bfs"
    DIJKSTRA = "dijkstra"
    ASTAR = "astar"

@dataclass
class ModelConfig:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"  # must be "cuda" for gpu
    cache_dir: str = "./cache/models"
    batch_size: int = 32

@dataclass
class GraphConfig:
    graph_type: GraphType = GraphType.THRESHOLD_BASED
    threshold: float = 0.25
    top_k: int = 8
    save_graph: bool = True
    graph_format: str = "adjacency"  # "adjacency" or "networkx"

@dataclass
class AlgorithmConfig:
    default_algorithm: Algorithm = Algorithm.DIJKSTRA
    heuristic_weight: float = 1.0 # for A*
    timeout_seconds: int = 30
    enable_cache: bool = True

@dataclass
class EvaluationConfig:
    test_cases_count: int = 10
    performance_iterations: int = 5
    save_results: bool = True
    plot_format: str = "png" 

@dataclass
class ProjectConfig:
    project_name: str = "Semantic Graph Search Algorithm"
    group_members: list = None
    version: str = "1.0.0"
    
    data_dir: str = "./data"
    results_dir: str = "./results"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    
    def __post_init__(self):
        if self.group_members is None:
            self.group_members = ["Sarina NaserMoghadasi","Hanane Ahmadi"]
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

DEFAULT_CONFIG = {
    "model": ModelConfig(),
    "graph": GraphConfig(),
    "algorithm": AlgorithmConfig(),
    "evaluation": EvaluationConfig(),
    "project": ProjectConfig()
}