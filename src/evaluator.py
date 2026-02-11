import time
import json
import os
from typing import List, Dict
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
import logging
from datetime import datetime

from config import Algorithm

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    algorithm: str
    avg_time: float
    std_time: float
    avg_distance: float
    std_distance: float
    avg_nodes_visited: float
    success_rate: float
    avg_path_length: float
    speedup_vs_dijkstra: float

class SystemEvaluator:
    
    def __init__(self, search_algorithms, config):
    
        self.algorithms = search_algorithms
        self.config = config
        self.results_dir = config.project.results_dir
        self.test_cases = []
        
        os.makedirs(os.path.join(self.results_dir, "performance"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "reports"), exist_ok=True)
    
    def load_test_cases(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.test_cases = json.load(f)
            logger.info(f"{len(self.test_cases)} random cases loaded")
        except Exception as e:
            logger.error(f"error in loading tests: {e}")
            self.test_cases = []
    
    def generate_random_test_cases(self, phrases: List[str], num_cases: int = 20):
        import random
        
        self.test_cases = []
        n = len(phrases)
        
        for _ in range(num_cases):
            start_idx = random.randint(0, n - 1)
            end_idx = random.randint(0, n - 1)
            
            while end_idx == start_idx:
                end_idx = random.randint(0, n - 1)
            
            self.test_cases.append({
                'start': phrases[start_idx],
                'end': phrases[end_idx],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'category': self._categorize_test(start_idx, end_idx, phrases)
            })
        
        logger.info(f"{num_cases} random cases generated")
        
        self._save_test_cases()
    
    def _categorize_test(self, start_idx: int, end_idx: int, phrases: List[str]) -> str:
        return "general"
    
    def _save_test_cases(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"test_cases_{timestamp}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_cases, f, ensure_ascii=False, indent=2)
        
        logger.info(f"test saved in: {filepath}")
    
    def run_performance_evaluation(self) -> Dict[str, PerformanceMetrics]:
       
        if not self.test_cases:
            logger.error("no tests available")
            return {}
        
        logger.info("evaluation started...")
        logger.info(f"number of tests: {len(self.test_cases)}")
        logger.info(f"number of repeats: {self.config.performance_iterations}")
        
        results = {algo.value: [] for algo in Algorithm}
        
        for test_case in tqdm(self.test_cases, desc="tests evaluation"):
            start_phrase = test_case['start']
            end_phrase = test_case['end']
            
            for algo in Algorithm:
                algo_results = []
                
                for _ in range(self.config.performance_iterations):
                    result = self.algorithms.find_path(start_phrase, end_phrase, algo)
                    algo_results.append(result)
                
                results[algo.value].append(algo_results)
        
        metrics = self._analyze_results(results)
        
        if self.config.save_results:
            self._save_performance_results(metrics)
            self._generate_performance_plots(metrics)
        
        return metrics
    
    def _analyze_results(self, results: Dict) -> Dict[str, PerformanceMetrics]:
        metrics = {}
        
        for algo_name, algo_results in results.items():
            times = []
            distances = []
            nodes_visited = []
            path_lengths = []
            successes = 0
            total = 0
            
            for test_results in algo_results:
                for result in test_results:
                    total += 1
                    
                    if result.success:
                        successes += 1
                        times.append(result.execution_time)
                        distances.append(result.total_distance)
                        nodes_visited.append(result.nodes_visited)
                        path_lengths.append(len(result.path))
            
            if successes > 0:
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                avg_nodes = np.mean(nodes_visited)
                avg_path_len = np.mean(path_lengths)
                success_rate = successes / total
            else:
                avg_time = std_time = avg_distance = std_distance = 0
                avg_nodes = avg_path_len = 0
                success_rate = 0
            
            metrics[algo_name] = PerformanceMetrics(
                algorithm=algo_name,
                avg_time=avg_time,
                std_time=std_time,
                avg_distance=avg_distance,
                std_distance=std_distance,
                avg_nodes_visited=avg_nodes,
                success_rate=success_rate,
                avg_path_length=avg_path_len,
                speedup_vs_dijkstra=0  # calculated later
            )
        
        if 'dijkstra' in metrics and metrics['dijkstra'].avg_time > 0:
            dijkstra_time = metrics['dijkstra'].avg_time
            
            for algo_name, metric in metrics.items():
                if algo_name != 'dijkstra' and metric.avg_time > 0:
                    metric.speedup_vs_dijkstra = dijkstra_time / metric.avg_time
        
        return metrics
    
    def _save_performance_results(self, metrics: Dict[str, PerformanceMetrics]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # saving data in json
        json_data = {}
        for algo_name, metric in metrics.items():
            json_data[algo_name] = {
                'algorithm': metric.algorithm,
                'avg_time': metric.avg_time,
                'std_time': metric.std_time,
                'avg_distance': metric.avg_distance,
                'std_distance': metric.std_distance,
                'avg_nodes_visited': metric.avg_nodes_visited,
                'success_rate': metric.success_rate,
                'avg_path_length': metric.avg_path_length,
                'speedup_vs_dijkstra': metric.speedup_vs_dijkstra
            }
        
        json_path = os.path.join(self.results_dir, "performance", f"metrics_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # saving in csv format
        csv_data = []
        for algo_name, metric in metrics.items():
            csv_data.append({
                'Algorithm': metric.algorithm,
                'Avg Time (s)': f"{metric.avg_time:.6f}",
                'Std Time': f"{metric.std_time:.6f}",
                'Avg Distance': f"{metric.avg_distance:.4f}",
                'Std Distance': f"{metric.std_distance:.4f}",
                'Avg Nodes Visited': f"{metric.avg_nodes_visited:.1f}",
                'Success Rate': f"{metric.success_rate:.2%}",
                'Avg Path Length': f"{metric.avg_path_length:.1f}",
                'Speedup vs Dijkstra': f"{metric.speedup_vs_dijkstra:.2f}x"
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.results_dir, "performance", f"metrics_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"results saved in {json_path} and {csv_path}")
    
    def _generate_performance_plots(self, metrics: Dict[str, PerformanceMetrics]):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            plt.rcParams['font.family'] = 'DejaVu Sans'  # adding Persian
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. time
            algorithms = list(metrics.keys())
            avg_times = [metrics[algo].avg_time for algo in algorithms]
            std_times = [metrics[algo].std_time for algo in algorithms]
            
            axes[0, 0].bar(algorithms, avg_times, yerr=std_times, capsize=5)
            axes[0, 0].set_title('Average Algorithm Execution Time')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. success rate
            success_rates = [metrics[algo].success_rate for algo in algorithms]
            axes[0, 1].bar(algorithms, success_rates)
            axes[0, 1].set_title('Algorithm Success Rate')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. nodes visited
            nodes_visited = [metrics[algo].avg_nodes_visited for algo in algorithms]
            axes[1, 0].bar(algorithms, nodes_visited)
            axes[1, 0].set_title('Average Visited Nodes')
            axes[1, 0].set_ylabel('Number of Nodes')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. speedup vs dijkstra 
            speedups = [metrics[algo].speedup_vs_dijkstra for algo in algorithms]
            axes[1, 1].bar(algorithms, speedups)
            axes[1, 1].set_title('Speedup vs Dijkstra')
            axes[1, 1].set_ylabel('Speedup Factor')
            axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, "performance", f"plots_{timestamp}.{self.config.plot_format}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"plots saved in: {plot_path}")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available")
    
    def evaluate_scalability(self, max_nodes: int = 100, step: int = 10):
        import random
        import string
        
        logger.info("scalability started...")
        
        scalability_results = []
        
        for n in range(step, max_nodes + 1, step):
            phrases = []
            for i in range(n):
                random_str = ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(5, 15)))
                phrases.append(f"word_{i}_{random_str}")
            
            from .graph_builder import GraphBuilder
            from .semantic_model import SemanticModel
            
            model_config = type('obj', (object,), {
                'model_name': 'all-MiniLM-L6-v2',
                'device': 'cpu',
                'cache_dir': './cache',
                'batch_size': 32,
                'enable_cache': False
            })()
            
            model = SemanticModel(model_config)
            graph_config = type('obj', (object,), {
                'graph_type': 'threshold_based',
                'threshold': 0.3,
                'top_k': 5,
                'save_graph': False,
                'project': type('obj', (object,), {
                    'results_dir': './results'
                })()
            })()
            
            builder = GraphBuilder(model, graph_config)
            builder.build_graph(phrases)
            
            start_time = time.time()
            builder.build_graph(phrases)
            graph_build_time = time.time() - start_time
            
            search_times = []
            for _ in range(5): 
                start_idx = random.randint(0, n-1)
                end_idx = random.randint(0, n-1)
                
                start_time = time.time()
                builder.get_neighbors(start_idx)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = np.mean(search_times) if search_times else 0
            
            scalability_results.append({
                'num_nodes': n,
                'graph_build_time': graph_build_time,
                'avg_search_time': avg_search_time,
                'memory_usage': n**2 * 8 / (1024**2)  #MB
            })
            
            logger.info(f"n={n}: build={graph_build_time:.3f}s, search={avg_search_time:.3f}s")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scal_path = os.path.join(self.results_dir, "reports", f"scalability_{timestamp}.json")
        
        with open(scal_path, 'w', encoding='utf-8') as f:
            json.dump(scalability_results, f, indent=2)
        
        self._plot_scalability_results(scalability_results)
        
        logger.info(f"scalability results saved in {scal_path}")
    
    def _plot_scalability_results(self, results: List[Dict]):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            
            nodes = [r['num_nodes'] for r in results]
            build_times = [r['graph_build_time'] for r in results]
            search_times = [r['avg_search_time'] for r in results]
            memory_usage = [r['memory_usage'] for r in results]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].plot(nodes, build_times, 'o-', linewidth=2)
            axes[0].set_xlabel('Number of Nodes')
            axes[0].set_ylabel('Graph Build Time (seconds)')
            axes[0].set_title('Graph Build Time Scalability')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(nodes, search_times, 's-', linewidth=2, color='green')
            axes[1].set_xlabel('Number of Nodes')
            axes[1].set_ylabel('Search Time (seconds)')
            axes[1].set_title('Search Time Scalability')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(nodes, memory_usage, '^-', linewidth=2, color='red')
            axes[2].set_xlabel('Number of Nodes')
            axes[2].set_ylabel('Memory Usage (MB)')
            axes[2].set_title('Memory Usage Scalability')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, "reports", f"scalability_plot_{timestamp}.{self.config.plot_format}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib is not available")