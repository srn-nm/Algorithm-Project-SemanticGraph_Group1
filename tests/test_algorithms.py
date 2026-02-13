#unit test

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.graph_builder import GraphBuilder, GraphType
from src.algorithms import SearchAlgorithms, Algorithm

class TestAlgorithms(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        class SimpleModel:
            def compute_similarity(self, text1, text2):
                if text1 == text2:
                    return 1.0
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                if not words1 or not words2:
                    return 0.0
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                return intersection / union if union > 0 else 0.0
        
        cls.simple_model = SimpleModel()
        
        cls.test_phrases = [
            "سیب",
            "میوه", 
            "تکنولوژی",
            "آیفون",
            "گوشی هوشمند",
            "کامپیوتر",
            "لپ تاپ",
            "برنامه نویسی"
        ]
    
    def setUp(self):
        from config import GraphConfig, AlgorithmConfig, ProjectConfig
        
        class TestConfig:
            model = type('obj', (object,), {
                'model_name': 'test',
                'device': 'cpu',
                'cache_dir': './test_cache',
                'batch_size': 32,
                'enable_cache': False
            })()
            
            graph = GraphConfig(
                graph_type=GraphType.THRESHOLD_BASED,
                threshold=0.2,
                top_k=3,
                save_graph=False
            )
            
            algorithm = AlgorithmConfig(
                default_algorithm=Algorithm.DIJKSTRA,
                heuristic_weight=1.0,
                timeout_seconds=10,
                enable_cache=False
            )
            
            project = ProjectConfig(
                project_name="Test",
                group_members=["Test"],
                results_dir="./test_results"
            )
        
        self.config = TestConfig()
        
        self.graph_builder = GraphBuilder(self.simple_model, self.config.graph)
        self.graph = self.graph_builder.build_graph(self.test_phrases)
        
        self.algorithms = SearchAlgorithms(
            self.graph_builder,
            self.config.algorithm,
            self.simple_model
        )
    
    def test_bfs_simple_path(self):
        result = self.algorithms.find_path("سیب", "میوه", Algorithm.BFS)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.path), 0)
        self.assertIn("سیب", result.path_phrases)
        self.assertIn("میوه", result.path_phrases)
    
    def test_dijkstra_shortest_path(self):
        result = self.algorithms.find_path("سیب", "آیفون", Algorithm.DIJKSTRA)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.path), 0)
        self.assertGreater(result.total_distance, 0)
    
    def test_astar_with_heuristic(self):
        result = self.algorithms.find_path("سیب", "آیفون", Algorithm.ASTAR)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.path), 0)
        self.assertGreater(result.total_distance, 0)
    
    def test_path_not_found(self):
        isolated_phrases = self.test_phrases + ["واژه_منزوی"]
        
        graph_builder = GraphBuilder(self.simple_model, self.config.graph)
        graph_builder.build_graph(isolated_phrases)
        
        algorithms = SearchAlgorithms(
            graph_builder,
            self.config.algorithm,
            self.simple_model
        )
        
        result = algorithms.find_path("واژه_منزوی", "سیب", Algorithm.DIJKSTRA)
        
        self.assertFalse(result.success)
        self.assertIn("مسیری", result.error_message)
    
    def test_algorithm_comparison(self):
        results = self.algorithms.compare_algorithms("سیب", "آیفون")
        
        self.assertIn("bfs", results)
        self.assertIn("dijkstra", results)
        self.assertIn("astar", results)
        
        success_count = sum(1 for r in results.values() if r.success)
        self.assertGreater(success_count, 0)
    
    def test_k_shortest_paths(self):
        paths = self.algorithms.find_k_shortest_paths(
            self.graph_builder.phrase_to_idx["سیب"],
            self.graph_builder.phrase_to_idx["آیفون"],
            k=2
        )
        
        self.assertLessEqual(len(paths), 2)
        if paths:
            for path_result in paths:
                self.assertTrue(path_result.success)
                self.assertGreater(len(path_result.path), 0)
    
    def test_edge_details(self):
        result = self.algorithms.find_path("سیب", "میوه", Algorithm.DIJKSTRA)
        
        if result.success:
            self.assertGreater(len(result.edge_details), 0)
            for edge in result.edge_details:
                self.assertIn('from', edge)
                self.assertIn('to', edge)
                self.assertIn('weight', edge)
                self.assertIn('similarity', edge)
                self.assertGreaterEqual(edge['similarity'], 0)
                self.assertLessEqual(edge['similarity'], 1)

if __name__ == '__main__':
    unittest.main()