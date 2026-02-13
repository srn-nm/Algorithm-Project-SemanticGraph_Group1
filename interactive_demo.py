import sys
import os
import numpy as np
from typing import List, Dict
from colorama import init, Fore, Style
import networkx as nx
from src.semantic_model import SemanticModel
from src.graph_builder import GraphBuilder, GraphType
from src.algorithms import SearchAlgorithms, Algorithm
from config import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


init(autoreset=True)

class InteractiveDemo:
    
    def __init__(self):
        self.system = None
        self.phrases = []
    
    def setup_system(self):
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN} Semantic Graph Search System - Interactive Demo")
        print(f"{Fore.CYAN}{'='*60}")
        
        print(f"\n{Fore.YELLOW} Select semantic model:")
        print(f"{Fore.WHITE}1. MiniLM (small and fast)")
        print(f"{Fore.WHITE}2. Multilingual MiniLM (for Persian)")
        print(f"{Fore.WHITE}3. MPNet (higher accuracy)")
        
        model_choice = input(f"\n{Fore.GREEN}Your choice (1-3): ").strip()
        
        model_map = {
            '1': 'all-MiniLM-L6-v2',
            '2': 'paraphrase-multilingual-MiniLM-L12-v2',
            '3': 'all-mpnet-base-v2'
        }
        
        model_name = model_map.get(model_choice, 'paraphrase-multilingual-MiniLM-L12-v2')
        
        print(f"\n{Fore.YELLOW} Select graph construction method:")
        print(f"{Fore.WHITE}1. Fully connected graph")
        print(f"{Fore.WHITE}2. Threshold-based graph (recommended)")
        print(f"{Fore.WHITE}3. Top-K graph")
        
        graph_choice = input(f"\n{Fore.GREEN}Your choice (1-3): ").strip()
        
        graph_type_map = {
            '1': GraphType.FULLY_CONNECTED,
            '2': GraphType.THRESHOLD_BASED,
            '3': GraphType.TOP_K
        }
        
        graph_type = graph_type_map.get(graph_choice, GraphType.THRESHOLD_BASED)
        
        threshold = 0.25
        top_k = 5
        
        if graph_type == GraphType.THRESHOLD_BASED:
            threshold_input = input(f"\n{Fore.GREEN}Similarity threshold (0.1-0.9) [0.25]: ").strip()
            if threshold_input:
                threshold = float(threshold_input)
        
        elif graph_type == GraphType.TOP_K:
            k_input = input(f"\n{Fore.GREEN}Number of top neighbors (K) [5]: ").strip()
            if k_input:
                top_k = int(k_input)
        
        self.load_phrases()
        
        config = DEFAULT_CONFIG.copy()
        config['model'].model_name = model_name
        config['graph'].graph_type = graph_type
        config['graph'].threshold = threshold
        config['graph'].top_k = top_k
        
        from main import SemanticGraphSearchSystem
        self.system = SemanticGraphSearchSystem(config)
        self.system.phrases = self.phrases
        
        print(f"\n{Fore.GREEN} System initialized successfully!")
    
    def load_phrases(self):
        print(f"\n{Fore.YELLOW} Load phrases:")
        print(f"{Fore.WHITE}1. Use default phrases")
        print(f"{Fore.WHITE}2. Enter phrases manually")
        print(f"{Fore.WHITE}3. Load from file")
        
        choice = input(f"\n{Fore.GREEN}Your choice (1-3): ").strip()
        
        if choice == '1':
            self.phrases = [
                "Artificial Intelligence", "Machine Learning", "Neural Network",
                "Natural Language Processing", "Computer Vision", "Data Mining",
                "Genetic Algorithm", "Robotics", "Automation", "Data Analysis",
                "Apple", "Fruit", "Technology", "iPhone", "Smartphone",
                "Computer", "Laptop", "Programming", "Algorithm", "Data Structure"
            ]
        
        elif choice == '2':
            print(f"\n{Fore.YELLOW}Enter phrases (empty to finish):")
            self.phrases = []
            i = 1
            while True:
                phrase = input(f"{Fore.WHITE}Phrase {i}: ").strip()
                if not phrase:
                    break
                self.phrases.append(phrase)
                i += 1
        
        elif choice == '3':
            filepath = input(f"\n{Fore.GREEN}File path: ").strip()
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.phrases = [line.strip() for line in f if line.strip()]
            else:
                print(f"{Fore.RED} File does not exist!")
                self.load_phrases()
                return
        
        print(f"\n{Fore.GREEN} {len(self.phrases)} phrases loaded")
    
    def run_demo(self):
        if not self.system:
            self.setup_system()
        
        print(f"\n{Fore.YELLOW} Building semantic graph...")
        self.system.build_graph()
        
        while True:
            self.show_menu()
            choice = input(f"\n{Fore.GREEN}Your choice: ").strip()
            
            if choice == '1':
                self.search_path()
            elif choice == '2':
                self.compare_algorithms()
            elif choice == '3':
                self.show_graph_info()
            elif choice == '4':
                self.visualize_graph()
            elif choice == '5':
                self.test_similarity()
            elif choice == '6':
                self.performance_evaluation()
            elif choice == '7':
                print(f"\n{Fore.CYAN} Goodbye!")
                break
            else:
                print(f"{Fore.RED} Invalid choice!")
    
    def show_menu(self):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN} Main Menu")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.WHITE}1. Search path")
        print(f"{Fore.WHITE}2. Compare algorithms")
        print(f"{Fore.WHITE}3. Graph information")
        print(f"{Fore.WHITE}4. Visualize graph")
        print(f"{Fore.WHITE}5. Test similarity")
        print(f"{Fore.WHITE}6. Performance evaluation")
        print(f"{Fore.WHITE}7. Exit")
    
    def search_path(self):
        print("\n" + "="*60)
        print(" Semantic path search")
        print("="*60)

        print("\nAvailable phrases:")
        for i, phrase in enumerate(self.system.phrases, 1):
            print(f"{i:2}. {phrase}")

        start_input = input("\nStart phrase number: ").strip()
        if not start_input.isdigit():
            print(" Error: Please enter a valid number")
            return

        end_input = input("End phrase number: ").strip()
        if not end_input.isdigit():
            print(" Error: Please enter a valid number")
            return

        start_idx = int(start_input) - 1
        end_idx = int(end_input) - 1

        if start_idx < 0 or start_idx >= len(self.system.phrases):
            print(f" Error: Invalid start number. Please enter 1-{len(self.system.phrases)}")
            return
        
        if end_idx < 0 or end_idx >= len(self.system.phrases):
            print(f" Error: Invalid end number. Please enter 1-{len(self.system.phrases)}")
            return

        start_phrase = self.system.phrases[start_idx]
        end_phrase = self.system.phrases[end_idx]
        
        print(f"\n Selected: '{start_phrase}' → '{end_phrase}'")
        print("\nSearch algorithm:")
        print("1. BFS (unweighted)")
        print("2. Dijkstra (shortest path)")
        print("3. A* (with heuristic)")
        
        algo_choice = input("Your choice (1-3): ").strip()
        if not algo_choice.isdigit() or algo_choice not in ['1', '2', '3']:
            print(" Invalid choice. Using Dijkstra.")
            from config import Algorithm
            algorithm = Algorithm.DIJKSTRA
        else:
            from config import Algorithm
            algorithm_map = {'1': Algorithm.BFS, '2': Algorithm.DIJKSTRA, '3': Algorithm.ASTAR}
            algorithm = algorithm_map[algo_choice]
        
        print("\n Searching...")

        result = self.system.find_path(start_phrase, end_phrase, algorithm)

        self._display_search_results(result)

    def _display_search_results(self, result):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN} Search Results")
        print(f"{Fore.CYAN}{'='*60}")
        
        if isinstance(result, dict):
            if result.get('success'):
                print(f"{Fore.GREEN}✓ Path found!")
                print(f"\n{Fore.WHITE}Algorithm: {result.get('algorithm', 'Unknown').upper()}")

                if 'path_phrases' in result and result['path_phrases']:
                    path_str = " → ".join(result['path_phrases'])
                    print(f"\n{Fore.YELLOW}Path:")
                    print(f"{Fore.WHITE}{path_str}")

                print(f"\n{Fore.YELLOW}Metrics:")
                print(f"{Fore.WHITE}  Total distance: {result.get('total_distance', 0):.4f}")
                print(f"{Fore.WHITE}  Execution time: {result.get('execution_time', 0):.4f} seconds")
                print(f"{Fore.WHITE}  Nodes visited: {result.get('nodes_visited', 0)}")

                if 'edge_details' in result and result['edge_details']:
                    print(f"\n{Fore.YELLOW}Edge details:")
                    for i, edge in enumerate(result['edge_details'][:5]):
                        print(f"{Fore.WHITE}  {i+1}. {edge['from']} → {edge['to']}: "
                            f"similarity={edge['similarity']:.3f}, distance={edge['distance']:.3f}")
                    if len(result['edge_details']) > 5:
                        print(f"{Fore.WHITE}  ... and {len(result['edge_details']) - 5} more edges")
            else:
                print(f"{Fore.RED}✗ Path not found!")
                if 'error_message' in result and result['error_message']:
                    print(f"{Fore.RED}  Error: {result['error_message']}")
                else:
                    print(f"{Fore.RED}  No path exists between these phrases")
        
        elif hasattr(result, 'success'):
            if result.success:
                print(f"{Fore.GREEN}✓ Path found!")
                print(f"\n{Fore.WHITE}Algorithm: {result.algorithm.upper()}")

                if hasattr(result, 'path_phrases') and result.path_phrases:
                    path_str = " → ".join(result.path_phrases)
                    print(f"\n{Fore.YELLOW}Path:")
                    print(f"{Fore.WHITE}{path_str}")

                print(f"\n{Fore.YELLOW}Metrics:")
                print(f"{Fore.WHITE}  Total distance: {result.total_distance:.4f}")
                print(f"{Fore.WHITE}  Execution time: {result.execution_time:.4f} seconds")
                print(f"{Fore.WHITE}  Nodes visited: {result.nodes_visited}")

                if hasattr(result, 'edge_details') and result.edge_details:
                    print(f"\n{Fore.YELLOW}Edge details:")
                    for i, edge in enumerate(result.edge_details[:5]):
                        print(f"{Fore.WHITE}  {i+1}. {edge['from']} → {edge['to']}: "
                            f"similarity={edge['similarity']:.3f}, distance={edge['distance']:.3f}")
                    if len(result.edge_details) > 5:
                        print(f"{Fore.WHITE}  ... and {len(result.edge_details) - 5} more edges")
            else:
                print(f"{Fore.RED}✗ Path not found!")
                if hasattr(result, 'error_message') and result.error_message:
                    print(f"{Fore.RED}  Error: {result.error_message}")
                else:
                    print(f"{Fore.RED}  No path exists between these phrases")
        else:
            print(f"{Fore.RED} Invalid result format")
        
        print(f"{Fore.CYAN}{'='*60}")
    
    def compare_algorithms(self):
        print(f"\n{Fore.YELLOW} Compare algorithms")
        print("="*60)

        print("\nAvailable phrases:")
        for i, phrase in enumerate(self.system.phrases, 1):
            print(f"{i:2}. {phrase}")

        start_input = input(f"\n{Fore.GREEN}Start phrase number: ").strip()
        if not start_input.isdigit():
            print(f"{Fore.RED} Error: Please enter a valid number")
            return

        end_input = input(f"{Fore.GREEN}End phrase number: ").strip()
        if not end_input.isdigit():
            print(f"{Fore.RED} Error: Please enter a valid number")
            return

        start_idx = int(start_input) - 1
        end_idx = int(end_input) - 1

        if start_idx < 0 or start_idx >= len(self.system.phrases):
            print(f"{Fore.RED} Error: Invalid start number. Please enter 1-{len(self.system.phrases)}")
            return
        
        if end_idx < 0 or end_idx >= len(self.system.phrases):
            print(f"{Fore.RED} Error: Invalid end number. Please enter 1-{len(self.system.phrases)}")
            return

        start_phrase = self.system.phrases[start_idx]
        end_phrase = self.system.phrases[end_idx]
        
        print(f"\n Selected: '{start_phrase}' → '{end_phrase}'")
        
        print(f"\n{Fore.YELLOW} Comparing algorithms...")
        results = self.system.compare_algorithms(start_phrase, end_phrase)
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN} Comparison Results")
        print(f"{Fore.CYAN}{'='*60}")
        
        for algo_name, result in results.items():
            print(f"\n{Fore.YELLOW}{algo_name.upper()}:")
            
            if result.get('success'):
                print(f"{Fore.GREEN}  ✓ Success")
                if 'path_phrases' in result and result['path_phrases']:
                    path_preview = " → ".join(result['path_phrases'][:3])
                    if len(result['path_phrases']) > 3:
                        path_preview += " → ..."
                    print(f"{Fore.WHITE}  Path: {path_preview}")
                print(f"{Fore.WHITE}  Distance: {result.get('total_distance', 0):.4f}")
                print(f"{Fore.WHITE}  Time: {result.get('execution_time', 0):.4f}s")
                print(f"{Fore.WHITE}  Nodes visited: {result.get('nodes_visited', 0)}")
            else:
                print(f"{Fore.RED}  ✗ Failed: {result.get('error_message', 'Unknown error')}")
        
    def show_graph_info(self):
        if not self.system.graph:
            print(f"{Fore.RED} Build the graph first")
            return
        
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN} Graph Information")
        print(f"{Fore.CYAN}{'='*60}")
        
        graph = self.system.graph
        print(f"{Fore.WHITE}Number of nodes: {graph.number_of_nodes()}")
        print(f"{Fore.WHITE}Number of edges: {graph.number_of_edges()}")
        
        density = nx.density(graph)
        print(f"{Fore.WHITE}Density: {density:.4f}")
        
        degrees = [d for _, d in graph.degree()]
        print(f"{Fore.WHITE}Average degree: {np.mean(degrees):.2f}")
        print(f"{Fore.WHITE}Maximum degree: {max(degrees)}")
        print(f"{Fore.WHITE}Minimum degree: {min(degrees)}")
        
        weights = [data['weight'] for _, _, data in graph.edges(data=True)]
        print(f"{Fore.WHITE}Average edge weight: {np.mean(weights):.4f}")
        print(f"{Fore.WHITE}Maximum weight: {max(weights):.4f}")
        print(f"{Fore.WHITE}Minimum weight: {min(weights):.4f}")
        
        is_connected = nx.is_connected(graph)
        print(f"{Fore.WHITE}Connected: {'Yes' if is_connected else 'No'}")
        
        if not is_connected:
            components = nx.number_connected_components(graph)
            print(f"{Fore.WHITE}Number of components: {components}")
    
    def visualize_graph(self):
        if not self.system.graph:
            print(f"{Fore.RED} Build the graph first")
            return
        
        print(f"\n{Fore.YELLOW} Visualizing graph...")
        self.system.visualize_graph()
        print(f"{Fore.GREEN} Graph visualization completed. Results saved in results/visualizations/")
    
    def test_similarity(self):
        print(f"\n{Fore.YELLOW} Similarity computation test")
        
        while True:
            text1 = input(f"\n{Fore.GREEN}First text (empty to return): ").strip()
            if not text1:
                break
            text2 = input(f"{Fore.GREEN}Second text: ").strip()
            if not text2:
                break
            
            result = self.system.semantic_model.compute_similarity(text1, text2)
            print(f"\n{Fore.CYAN}Results:")
            print(f"{Fore.WHITE}Similarity: {result.similarity:.4f}")
            print(f"{Fore.WHITE}Distance: {result.distance:.4f}")
            print(f"{Fore.WHITE}Cached: {'Yes' if result.cached else 'No'}")
    
    def performance_evaluation(self):
        if not self.system.graph:
            print(f"{Fore.RED} Build the graph first")
            return
        
        print(f"\n{Fore.YELLOW} Running performance evaluation...")
        print(f"{Fore.WHITE}This may take a few moments...")
        
        metrics = self.system.run_performance_evaluation(num_test_cases=10)
        
        if metrics:
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"{Fore.CYAN} Performance Evaluation Results")
            print(f"{Fore.CYAN}{'='*60}")
            
            for algo_name, metric in metrics.items():
                print(f"\n{Fore.YELLOW}{algo_name.upper()}:")
                print(f"{Fore.WHITE}  Average time: {metric.avg_time:.6f}s")
                print(f"{Fore.WHITE}  Average distance: {metric.avg_distance:.4f}")
                print(f"{Fore.WHITE}  Nodes visited: {metric.avg_nodes_visited:.1f}")
                print(f"{Fore.WHITE}  Success rate: {metric.success_rate:.2%}")
                if metric.speedup_vs_dijkstra > 0:
                    print(f"{Fore.WHITE}  Speedup vs Dijkstra: {metric.speedup_vs_dijkstra:.2f}x")
        else:
            print(f"{Fore.RED} Error during performance evaluation")

def main():
    demo = InteractiveDemo()
    demo.setup_system()
    demo.run_demo()

if __name__ == "__main__":
    main()
