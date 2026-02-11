import time
import json
import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
import logging
from datetime import datetime
import sys

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
        self.results_dir = config['project'].results_dir
        self.test_cases = []
        
        os.makedirs(os.path.join(self.results_dir, "performance"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "reports"), exist_ok=True)
    
    def load_test_cases(self, filepath: str):
        """Load test cases from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.test_cases = json.load(f)
            logger.info(f"{len(self.test_cases)} test cases loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Test cases file not found: {filepath}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test cases file: {e}")
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
    
    def generate_random_test_cases(self, phrases: List[str], num_cases: int = 20):
        """Generate random test cases from phrases"""
        import random
        
        if len(phrases) < 2:
            logger.error("Need at least 2 phrases to generate test cases")
            return
        
        self.test_cases = []
        n = len(phrases)
        
        for _ in range(num_cases):
            start_idx = random.randint(0, n - 1)
            end_idx = random.randint(0, n - 1)
            
            while end_idx == start_idx and n > 1:
                end_idx = random.randint(0, n - 1)
            
            self.test_cases.append({
                'start': phrases[start_idx],
                'end': phrases[end_idx],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'category': self._categorize_test(start_idx, end_idx, phrases)
            })
        
        logger.info(f"{num_cases} random test cases generated")
        self._save_test_cases()
    
    def _categorize_test(self, start_idx: int, end_idx: int, phrases: List[str]) -> str:
        """Categorize test case based on semantic similarity"""
        try:
            #  categorization logic must be implemented
            return "general"
        except Exception as e:
            logger.warning(f"Error categorizing test: {e}")
            return "uncategorized"
    
    def _save_test_cases(self):
        """Save generated test cases to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"test_cases_{timestamp}.json")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.test_cases, f, ensure_ascii=False, indent=2)
            logger.info(f"Test cases saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving test cases: {e}")
    
    def run_performance_evaluation(self) -> Dict[str, PerformanceMetrics]:
        """Run performance evaluation on all algorithms"""
        if not self.test_cases:
            logger.error("No test cases available for evaluation")
            return {}
        
        logger.info("Starting performance evaluation...")
        logger.info(f"Number of test cases: {len(self.test_cases)}")
        logger.info(f"Number of iterations per test: {self.config['evaluation'].performance_iterations}")
        
        results = {algo.value: [] for algo in Algorithm}
        
        for test_case in tqdm(self.test_cases, desc="Evaluating test cases"):
            start_phrase = test_case['start']
            end_phrase = test_case['end']
            
            for algo in Algorithm:
                algo_results = []
                
                for _ in range(self.config['evaluation'].performance_iterations):
                    try:
                        result = self.algorithms.find_path(start_phrase, end_phrase, algo)
                        algo_results.append(result)
                    except Exception as e:
                        logger.error(f"Error running {algo.value} on test case: {e}")
                        # Append a failed result
                        from dataclasses import dataclass
                        @dataclass
                        class FailedResult:
                            success: bool = False
                            execution_time: float = 0
                            total_distance: float = 0
                            nodes_visited: int = 0
                            path: List = None
                        algo_results.append(FailedResult())
                
                results[algo.value].append(algo_results)
        
        metrics = self._analyze_results(results)
        
        if self.config['evaluation'].save_results:
            self._save_performance_results(metrics)
            self._generate_performance_plots(metrics)
        
        return metrics
    
    def _analyze_results(self, results: Dict) -> Dict[str, PerformanceMetrics]:
        """Analyze raw results and compute performance metrics"""
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
                        path_lengths.append(len(result.path) if result.path else 0)
            
            if successes > 0:
                avg_time = float(np.mean(times))
                std_time = float(np.std(times))
                avg_distance = float(np.mean(distances))
                std_distance = float(np.std(distances))
                avg_nodes = float(np.mean(nodes_visited))
                avg_path_len = float(np.mean(path_lengths))
                success_rate = successes / total if total > 0 else 0
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
                speedup_vs_dijkstra=0.0
            )
        
        # Calculate speedup vs Dijkstra
        if 'dijkstra' in metrics and metrics['dijkstra'].avg_time > 0:
            dijkstra_time = metrics['dijkstra'].avg_time
            
            for algo_name, metric in metrics.items():
                if algo_name != 'dijkstra' and metric.avg_time > 0:
                    metric.speedup_vs_dijkstra = dijkstra_time / metric.avg_time
        
        return metrics
    
    def _save_performance_results(self, metrics: Dict[str, PerformanceMetrics]):
        """Save performance results to JSON and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
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
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON results: {e}")
        
        # Save as CSV
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
        
        try:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(self.results_dir, "performance", f"metrics_{timestamp}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Results saved to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving CSV results: {e}")
    
    def _generate_performance_plots(self, metrics: Dict[str, PerformanceMetrics]):
        """Generate performance visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            algorithms = list(metrics.keys())
            
            # 1. Execution Time
            avg_times = [metrics[algo].avg_time for algo in algorithms]
            std_times = [metrics[algo].std_time for algo in algorithms]
            
            bars1 = axes[0, 0].bar(algorithms, avg_times, yerr=std_times, 
                                   capsize=5, color='steelblue', alpha=0.8)
            axes[0, 0].set_title('Average Algorithm Execution Time', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars1, avg_times):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times)*0.01,
                               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 2. Success Rate
            success_rates = [metrics[algo].success_rate for algo in algorithms]
            bars2 = axes[0, 1].bar(algorithms, success_rates, color='seagreen', alpha=0.8)
            axes[0, 1].set_title('Algorithm Success Rate', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            for bar, val in zip(bars2, success_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{val:.1%}', ha='center', va='bottom', fontsize=9)
            
            # 3. Nodes Visited
            nodes_visited = [metrics[algo].avg_nodes_visited for algo in algorithms]
            bars3 = axes[1, 0].bar(algorithms, nodes_visited, color='darkorange', alpha=0.8)
            axes[1, 0].set_title('Average Visited Nodes', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Number of Nodes')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            for bar, val in zip(bars3, nodes_visited):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nodes_visited)*0.01,
                               f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Speedup vs Dijkstra
            speedups = [metrics[algo].speedup_vs_dijkstra for algo in algorithms]
            colors = ['crimson' if s < 1 else 'forestgreen' for s in speedups]
            bars4 = axes[1, 1].bar(algorithms, speedups, color=colors, alpha=0.8)
            axes[1, 1].set_title('Speedup vs Dijkstra', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Speedup Factor')
            axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Dijkstra baseline')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            for bar, val in zip(bars4, speedups):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.02,
                               f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, "performance", 
                                    f"plots_{timestamp}.{self.config['evaluation'].plot_format}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance plots saved to: {plot_path}")
            
        except ImportError as e:
            logger.warning(f"Matplotlib/Seaborn not available for plotting: {e}")
        except Exception as e:
            logger.error(f"Error generating performance plots: {e}")
    
    def evaluate_scalability(self, max_nodes: int = 100, step: int = 10):
        """Evaluate system scalability with increasing number of nodes"""
        import random
        import string
        import psutil
        import tracemalloc
        
        logger.info("Starting scalability evaluation...")
        
        # Import only when needed to avoid circular imports
        from src.graph_builder import GraphBuilder
        from src.semantic_model import SemanticModel
        
        scalability_results = []
        
        # Create base model config
        model_config = type('obj', (object,), {
            'model_name': self.config['model'].model_name,
            'device': self.config['model'].device,
            'cache_dir': self.config['model'].cache_dir,
            'batch_size': self.config['model'].batch_size,
            'enable_cache': False
        })()
        
        # Create base graph config
        graph_config = type('obj', (object,), {
            'graph_type': self.config['graph'].graph_type,
            'threshold': self.config['graph'].threshold,
            'top_k': self.config['graph'].top_k,
            'save_graph': False,
            'project': self.config['project']
        })()
        
        # Initialize model once and reuse
        model = SemanticModel(model_config)
        
        for n in range(step, max_nodes + 1, step):
            logger.info(f"Testing scalability with {n} nodes...")
            
            # Generate test phrases
            phrases = []
            base_terms = ["algorithm", "data", "machine", "learning", "neural", 
                         "network", "deep", "artificial", "intelligence", "system"]
            
            for i in range(n):
                base = random.choice(base_terms)
                suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
                phrases.append(f"{base}_{i}_{suffix}")
            
            # Build graph and measure performance
            builder = GraphBuilder(model, graph_config)
            
            # Measure graph build time
            start_time = time.time()
            graph = builder.build_graph(phrases)
            graph_build_time = time.time() - start_time
            
            # Measure memory usage
            import tracemalloc
            tracemalloc.start()
            # Force garbage collection
            import gc
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_usage_mb = peak / (1024 * 1024)
            
            # Measure search performance
            search_times = []
            if n > 1:
                for _ in range(min(5, n // 2)):  # Limit iterations for large n
                    start_idx = random.randint(0, n-1)
                    end_idx = random.randint(0, n-1)
                    while end_idx == start_idx and n > 1:
                        end_idx = random.randint(0, n-1)
                    
                    start_time = time.time()
                    # Test neighbor retrieval
                    builder.get_neighbors(start_idx)
                    search_time = time.time() - start_time
                    search_times.append(search_time)
            
            avg_search_time = np.mean(search_times) if search_times else 0
            
            # Estimate graph density
            num_edges = 0
            if hasattr(graph, 'edges'):
                num_edges = len(graph.edges())
            elif isinstance(graph, dict):
                for node in graph:
                    num_edges += len(graph.get(node, []))
                num_edges //= 2  # Undirected graph
            
            density = (2 * num_edges) / (n * (n - 1)) if n > 1 else 0
            
            scalability_results.append({
                'num_nodes': n,
                'graph_build_time': graph_build_time,
                'avg_search_time': avg_search_time,
                'memory_usage_mb': memory_usage_mb,
                'num_edges': num_edges,
                'density': density
            })
            
            logger.info(f"n={n}: build={graph_build_time:.3f}s, "
                       f"search={avg_search_time:.6f}s, "
                       f"memory={memory_usage_mb:.2f}MB, "
                       f"edges={num_edges}, density={density:.4f}")
        
        # Save scalability results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scal_path = os.path.join(self.results_dir, "reports", f"scalability_{timestamp}.json")
        
        try:
            with open(scal_path, 'w', encoding='utf-8') as f:
                json.dump(scalability_results, f, indent=2)
            logger.info(f"Scalability results saved to {scal_path}")
        except Exception as e:
            logger.error(f"Error saving scalability results: {e}")
        
        self._plot_scalability_results(scalability_results)
        
        return scalability_results
    
    def _plot_scalability_results(self, results: List[Dict]):
        """Generate scalability visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            
            nodes = [r['num_nodes'] for r in results]
            build_times = [r['graph_build_time'] for r in results]
            search_times = [r['avg_search_time'] for r in results]
            memory_usage = [r['memory_usage_mb'] for r in results]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Graph Build Time Scalability
            axes[0].plot(nodes, build_times, 'o-', linewidth=2, color='blue', markersize=6)
            axes[0].set_xlabel('Number of Nodes', fontsize=11)
            axes[0].set_ylabel('Graph Build Time (seconds)', fontsize=11)
            axes[0].set_title('Graph Build Time Scalability', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(nodes, build_times, 2)
            p = np.poly1d(z)
            axes[0].plot(nodes, p(nodes), '--', color='red', alpha=0.7, 
                        label=f'O(n²) trend: {z[0]:.2e}n²')
            axes[0].legend()
            
            # Search Time Scalability
            axes[1].plot(nodes, search_times, 's-', linewidth=2, color='green', markersize=6)
            axes[1].set_xlabel('Number of Nodes', fontsize=11)
            axes[1].set_ylabel('Search Time (seconds)', fontsize=11)
            axes[1].set_title('Search Time Scalability', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            if len(nodes) > 1:
                z_search = np.polyfit(nodes, search_times, 1)
                p_search = np.poly1d(z_search)
                axes[1].plot(nodes, p_search(nodes), '--', color='red', alpha=0.7,
                           label=f'O(n) trend: {z_search[0]:.2e}n')
                axes[1].legend()
            
            # Memory Usage Scalability
            axes[2].plot(nodes, memory_usage, '^-', linewidth=2, color='red', markersize=6)
            axes[2].set_xlabel('Number of Nodes', fontsize=11)
            axes[2].set_ylabel('Memory Usage (MB)', fontsize=11)
            axes[2].set_title('Memory Usage Scalability', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            # Add memory complexity annotation
            axes[2].annotate('O(n²) complexity', xy=(0.7, 0.9), xycoords='axes fraction',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor="yellow", alpha=0.2))
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, "reports", 
                                    f"scalability_plot_{timestamp}.{self.config['evaluation'].plot_format}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Scalability plots saved to: {plot_path}")
            
        except ImportError as e:
            logger.warning(f"Matplotlib/Seaborn not available for plotting: {e}")
        except Exception as e:
            logger.error(f"Error generating scalability plots: {e}")