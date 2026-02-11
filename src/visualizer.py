import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import plotly.graph_objects as go
import os
import datetime
import logging

logger = logging.getLogger(__name__)

class GraphVisualizer:
    
    def __init__(self, config):
        self.config = config
        self.results_dir = config.project.results_dir
        
        os.makedirs(os.path.join(self.results_dir, "visualizations"), exist_ok=True)
    
    def plot_graph(self, graph: nx.Graph, phrases: List[str], title: str = "semantic graph", save_path: Optional[str] = None):
        
        try:
            plt.figure(figsize=(12, 10))
            
            # layout
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
            
            node_colors = []
            for node in graph.nodes():
                degree = graph.degree(node)
                node_colors.append(degree)
            
            nodes = nx.draw_networkx_nodes(
                graph, pos,
                node_color=node_colors,
                node_size=300,
                cmap=plt.cm.viridis,
                alpha=0.8
            )
            
            edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
            edge_alphas = [1 - w/max(edge_weights) if edge_weights else 0.5 
                          for w in edge_weights]
            
            edges = nx.draw_networkx_edges(
                graph, pos,
                width=1,
                alpha=edge_alphas,
                edge_color='gray'
            )
            
            if len(graph.nodes()) <= 30:
                labels = {i: phrases[i] for i in graph.nodes()}
                nx.draw_networkx_labels(
                    graph, pos, 
                    labels=labels,
                    font_size=8,
                    font_family='DejaVu Sans'
                )
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.colorbar(nodes, label='node degree')
            plt.axis('off')
            
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.results_dir, 
                    "visualizations", 
                    f"graph_{timestamp}.png"
                )
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"graph saved in {save_path}")
            
        except Exception as e:
            logger.error(f"error in drawing graph: {e}")
    
    def plot_interactive_graph(self, graph: nx.Graph, phrases: List[str], title: str = "interactive semantic graph"):
       
        try:
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
            
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = graph[edge[0]][edge[1]]['weight']
                similarity = 1 - weight
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(f"similarity: {similarity:.3f}<br>distance: {weight:.3f}")
            
            node_x = [pos[node][0] for node in graph.nodes()]
            node_y = [pos[node][1] for node in graph.nodes()]
            
            node_degrees = [graph.degree(node) for node in graph.nodes()]
            node_text = [f"{phrases[node]}<br>degree: {graph.degree(node)}" 
                        for node in graph.nodes()]
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                mode='lines',
                text=edge_text * 3,
                hovertext=edge_text * 3
            )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[phrases[node][:15] + "..." if len(phrases[node]) > 15 else phrases[node] 
                      for node in graph.nodes()],
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    color=node_degrees,
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='node degree',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2
                ),
                textfont=dict(size=10)
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=title,
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.results_dir, 
                "visualizations", 
                f"interactive_graph_{timestamp}.html"
            )
            
            fig.write_html(save_path)
            logger.info(f"graph saved in {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Plotly is not available.")
            return None
    
    def plot_search_path(self, graph: nx.Graph, phrases: List[str], path: List[int], title: str = "founded path"):
        
        try:
            plt.figure(figsize=(12, 10))
            
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
            
            nx.draw_networkx_nodes(
                graph, pos,
                node_color='lightgray',
                node_size=100,
                alpha=0.3
            )
            
            nx.draw_networkx_edges(
                graph, pos,
                width=0.5,
                alpha=0.2,
                edge_color='gray'
            )
            
            if len(path) > 1:
                path_nodes = graph.subgraph(path)
                nx.draw_networkx_nodes(
                    path_nodes, pos,
                    node_color='red',
                    node_size=300,
                    alpha=0.8
                )
                
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=path_edges,
                    width=3,
                    alpha=0.8,
                    edge_color='red'
                )
                
                path_labels = {i: phrases[i] for i in path}
                nx.draw_networkx_labels(
                    graph, pos,
                    labels=path_labels,
                    font_size=10,
                    font_weight='bold',
                    font_family='DejaVu Sans'
                )
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.axis('off')
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.results_dir, 
                "visualizations", 
                f"path_{timestamp}.png"
            )
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"path plot saved in {save_path}")
            
        except Exception as e:
            logger.error(f"error in drawing the path: {e}")
    
    def plot_algorithm_comparison(self, metrics: Dict):
        try:
            import pandas as pd
            
            # convert to DataFrame
            data = []
            for algo_name, metric in metrics.items():
                data.append({
                    'Algorithm': algo_name,
                    'Average Time (seconds)': metric.avg_time,
                    'Average Distance': metric.avg_distance,
                    'Visited Nodes': metric.avg_nodes_visited,
                    'Success Rate': metric.success_rate,
                    'Speedup vs Dijkstra': metric.speedup_vs_dijkstra
                })
            
            df = pd.DataFrame(data)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. time
            axes[0, 0].bar(df['Algorithm'], df['Average Time (seconds)'])
            axes[0, 0].set_title('Algorithm Execution Time')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. distance
            axes[0, 1].bar(df['Algorithm'], df['Average Distance'])
            axes[0, 1].set_title('Found Path Distance')
            axes[0, 1].set_ylabel('Distance')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3: efficiency
            axes[1, 0].bar(df['Algorithm'], df['Visited Nodes'])
            axes[1, 0].set_title('Algorithm Efficiency')
            axes[1, 0].set_ylabel('Number of Visited Nodes')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. speed
            axes[1, 1].bar(df['Algorithm'], df['Speedup vs Dijkstra'])
            axes[1, 1].set_title('Relative Speed')
            axes[1, 1].set_ylabel('Speedup Factor')
            axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                self.results_dir, 
                "visualizations", 
                f"algorithm_comparison_{timestamp}.png"
            )
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"comparison chart saved in {save_path}")
            
        except Exception as e:
            logger.error(f"error in drawing comparison chart: {e}")