import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from config import *
from src.semantic_model import SemanticModel
from src.graph_builder import GraphBuilder
from src.algorithms import SearchAlgorithms
from src.evaluator import SystemEvaluator
from src.visualizer import GraphVisualizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src')) # adding src to path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/project_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SemanticGraphSearchSystem:
    
    def __init__(self, config: Optional[Dict] = None):
        
        if config is None:
            config = DEFAULT_CONFIG
        
        self.config = config
        
        logger.info(" Initializing Semantic Graph Search system...")
        
        self.semantic_model = SemanticModel(config['model'])
        self.graph_builder = GraphBuilder(self.semantic_model, config['graph'])
        self.search_algorithms = SearchAlgorithms(
            self.graph_builder, 
            config['algorithm'],
            self.semantic_model
        )
        self.evaluator = SystemEvaluator(self.search_algorithms, config)
        self.visualizer = GraphVisualizer(config['project'])
        
        self.phrases = []
        self.graph = None
        
        logger.info(" System initialized successfully")
    
    def load_phrases(self, filepath: str) -> List[str]:
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.txt'):
                    phrases = [line.strip() for line in f if line.strip()]
                elif filepath.endswith('.json'):
                    data = json.load(f)
                    phrases = data.get('phrases', [])
                else:
                    phrases = [line.strip() for line in f if line.strip()]
            
            self.phrases = phrases
            logger.info(f" {len(phrases)} phrases loaded from {filepath}")
            
            return phrases
            
        except Exception as e:
            logger.error(f"Error loading phrases: {e}")
            return []
    
    def build_graph(self, phrases: Optional[List[str]] = None):
      
        if phrases is None:
            phrases = self.phrases
        
        if not phrases:
            logger.error("No phrases available to build the graph")
            return
        
        logger.info(f" Starting graph construction with {len(phrases)} phrases...")
        
        self.graph = self.graph_builder.build_graph(phrases)
        
        logger.info(" Graph built successfully")
        
        return self.graph
    
    def find_path(self, start: str, end: str, 
                  algorithm: Algorithm = None) -> Dict:
        
        if self.graph is None:
            logger.error("Build the graph first")
            return {"error": "Graph not built"}
        
        result = self.search_algorithms.find_path(start, end, algorithm)
        
        if result.success:
            try:
                self.visualizer.plot_search_path(
                    self.graph,
                    self.phrases,
                    result.path,
                    title=f"Path from '{start}' to '{end}'"
                )
            except Exception as e:
                logger.warning(f"Error visualizing path: {e}")
        
        return self._format_result(result)
    
    def compare_algorithms(self, start: str, end: str) -> Dict:
        
        if self.graph is None:
            logger.error("Build the graph first")
            return {"error": "Graph not built"}
        
        results = self.search_algorithms.compare_algorithms(start, end)
        
        formatted = {}
        for algo_name, result in results.items():
            formatted[algo_name] = self._format_result(result)
        
        return formatted
    
    def run_performance_evaluation(self, num_test_cases: int = 20):
        
        if self.graph is None:
            logger.error("Build the graph first")
            return
        
        if not self.evaluator.test_cases:
            self.evaluator.generate_random_test_cases(self.phrases, num_test_cases)
        
        metrics = self.evaluator.run_performance_evaluation()
        
        try:
            self.visualizer.plot_algorithm_comparison(metrics)
        except Exception as e:
            logger.warning(f"Error visualizing results: {e}")
        
        return metrics
    
    def visualize_graph(self):
        if self.graph is None:
            logger.error("Build the graph first")
            return
        
        logger.info(" Visualizing graph...")
        
        self.visualizer.plot_graph(
            self.graph,
            self.phrases,
            title=f"Semantic Graph ({len(self.phrases)} nodes)"
        )
        
        self.visualizer.plot_interactive_graph(
            self.graph,
            self.phrases,
            title=f"Interactive Semantic Graph"
        )
    
    def run_scalability_test(self, max_nodes: int = 100):
      
        logger.info(" Starting scalability test...")
        
        self.evaluator.evaluate_scalability(max_nodes=max_nodes)
    
    def export_results(self, output_dir: str = None):
        
        if output_dir is None:
            output_dir = self.config['project'].results_dir
        
        logger.info(f" Saving results to {output_dir}...")
        
        system_info = {
            "project_name": self.config['project'].project_name,
            "group_members": self.config['project'].group_members,
            "version": self.config['project'].version,
            "model": self.config['model'].model_name,
            "graph_type": self.config['graph'].graph_type.value,
            "num_phrases": len(self.phrases),
            "timestamp": datetime.now().isoformat()
        }
        
        info_path = os.path.join(output_dir, "system_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(system_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f" System information saved in {info_path}")
    
    def _format_result(self, result) -> Dict:
        if hasattr(result, '__dict__'):
            return result.__dict__
        return result

def main():
    parser = argparse.ArgumentParser(
        description='Semantic Graph Search System - Algorithm Design Project',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--phrases', type=str, default='data/phrases/tech_phrases.txt', help='Path to phrases file')
    parser.add_argument('--start', type=str, help='Start phrase for search')
    parser.add_argument('--end', type=str, help='End phrase for search')
    parser.add_argument('--algorithm', type=str, choices=['bfs', 'dijkstra', 'astar'], default='dijkstra', help='Search algorithm')
    parser.add_argument('--build-graph', action='store_true', help='Build graph')
    parser.add_argument('--visualize', action='store_true', help='Visualize graph')
    parser.add_argument('--evaluate', action='store_true', help='Performance evaluation')
    parser.add_argument('--scalability', action='store_true', help='Scalability test')
    parser.add_argument('--compare', action='store_true', help='Compare algorithms')
    parser.add_argument('--config', type=str, help='JSON config file')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
            for key in config:
                if key in custom_config:
                    config[key].update(custom_config[key])
    
    system = SemanticGraphSearchSystem(config)
    
    if os.path.exists(args.phrases):
        system.load_phrases(args.phrases)
    else:
        logger.warning(f"Phrases file {args.phrases} does not exist. Using default phrases...")
        sample_phrases = [
            "Artificial Intelligence", "Machine Learning", "Neural Network",
            "Natural Language Processing", "Computer Vision", "Data Mining",
            "Genetic Algorithm", "Robotics", "Automation", "Data Analysis",
            "Apple", "Fruit", "Technology", "iPhone", "Smartphone",
            "Computer", "Laptop", "Programming", "Algorithm", "Data Structure"
        ]
        system.phrases = sample_phrases
    
    if args.build_graph or args.visualize or args.evaluate or args.compare:
        system.build_graph()
    
    if args.visualize:
        system.visualize_graph()
    
    if args.start and args.end:
        if args.compare:
            results = system.compare_algorithms(args.start, args.end)
            print("\n" + "="*60)
            print("Algorithm Comparison Results:")
            print("="*60)
            for algo, result in results.items():
                print(f"\n{algo.upper()}:")
                if result.get('success'):
                    print(f"  Path: {' → '.join(result['path_phrases'])}")
                    print(f"  Distance: {result['total_distance']:.4f}")
                    print(f"  Time: {result['execution_time']:.4f}s")
                    print(f"  Nodes visited: {result['nodes_visited']}")
                else:
                    print(f"  Error: {result.get('error_message', 'Unknown')}")
        else:
            from config import Algorithm
            algorithm = Algorithm(args.algorithm)
            result = system.find_path(args.start, args.end, algorithm)
            
            print("\n" + "="*60)
            print("Search Results:")
            print("="*60)
            if result.get('success'):
                print(f"Algorithm: {result['algorithm'].upper()}")
                print(f"Path: {' → '.join(result['path_phrases'])}")
                print(f"Total distance: {result['total_distance']:.4f}")
                print(f"Execution time: {result['execution_time']:.4f} seconds")
                print(f"Nodes visited: {result['nodes_visited']}")
                
                print("\nEdge details:")
                print("-"*40)
                for edge in result['edge_details']:
                    print(f"  {edge['from']} → {edge['to']}: "
                          f"distance={edge['distance']:.3f}, "
                          f"similarity={edge['similarity']:.3f}")
            else:
                print(f"Error: {result.get('error_message', 'Unknown')}")
    
    if args.evaluate:
        metrics = system.run_performance_evaluation()
        print("\n" + "="*60)
        print("Performance Evaluation Results:")
        print("="*60)
        for algo_name, metric in metrics.items():
            print(f"\n{algo_name.upper()}:")
            print(f"  Average time: {metric.avg_time:.6f}s (±{metric.std_time:.6f})")
            print(f"  Average distance: {metric.avg_distance:.4f} (±{metric.std_distance:.4f})")
            print(f"  Nodes visited: {metric.avg_nodes_visited:.1f}")
            print(f"  Success rate: {metric.success_rate:.2%}")
            print(f"  Speedup vs Dijkstra: {metric.speedup_vs_dijkstra:.2f}x")
    
    if args.scalability:
        system.run_scalability_test()
    
    system.export_results()
    
    print("\n" + "="*60)
    print(" Program executed successfully!")
    print("Results saved in results/ folder.")
    print("="*60)

if __name__ == "__main__":
    main()