import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any, Union
import plotly.graph_objects as go
import os
import datetime
import logging

logger = logging.getLogger(__name__)

class GraphVisualizer:
    
    def __init__(self, config):
        """
        Initialize the GraphVisualizer
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
        
        # Handle both object-style and dict-style config access
        if hasattr(config, 'project'):
            # Object-style access
            self.results_dir = config.project.results_dir
        elif isinstance(config, dict):
            # Dict-style access
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
            # Fallback
            self.results_dir = './results'
        
        # Create visualizations directory
        viz_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        logger.info(f"Visualizations will be saved to: {viz_dir}")
    
    def plot_graph(self, graph: nx.Graph, phrases: List[str], 
                   title: str = "Semantic Graph", 
                   save_path: Optional[str] = None,
                   max_labels: int = 30):
        """
        Plot static visualization of the semantic graph
        
        Args:
            graph: NetworkX graph object
            phrases: List of node labels
            title: Plot title
            save_path: Optional custom save path
            max_labels: Maximum number of labels to show
        """
        try:
            # Check if graph is empty
            if graph is None or len(graph.nodes()) == 0:
                logger.warning("Empty graph - nothing to visualize")
                return
            
            plt.figure(figsize=(12, 10))
            
            # Generate layout
            if len(graph.nodes()) > 100:
                pos = nx.spring_layout(graph, k=2, iterations=30, seed=42)
            else:
                pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
            
            # Node colors based on degree
            node_colors = []
            for node in graph.nodes():
                degree = graph.degree(node)
                node_colors.append(degree)
            
            # Node sizes based on degree
            node_sizes = [200 + 20 * graph.degree(node) for node in graph.nodes()]
            
            # Draw nodes
            nodes = nx.draw_networkx_nodes(
                graph, pos,
                node_color=node_colors,
                node_size=node_sizes,
                cmap=plt.cm.viridis,
                alpha=0.8,
                vmin=min(node_colors) if node_colors else 0,
                vmax=max(node_colors) if node_colors else 1
            )
            
            # Draw edges if they exist
            if graph.edges():
                edge_weights = []
                for u, v in graph.edges():
                    # Get weight, default to 0.5 if not present
                    weight = graph[u][v].get('weight', 0.5)
                    edge_weights.append(weight)
                
                if edge_weights:
                    max_weight = max(edge_weights)
                    edge_alphas = []
                    for w in edge_weights:
                        if max_weight > 0:
                            alpha = 1 - (w / max_weight) * 0.7
                        else:
                            alpha = 0.5
                        edge_alphas.append(alpha)
                    
                    nx.draw_networkx_edges(
                        graph, pos,
                        width=1,
                        alpha=edge_alphas,
                        edge_color='gray'
                    )
            
            # Draw labels for small to medium graphs
            if len(graph.nodes()) <= max_labels:
                labels = {}
                for i in graph.nodes():
                    if i < len(phrases):
                        # Truncate long phrases
                        label = phrases[i]
                        if len(label) > 20:
                            label = label[:17] + "..."
                        labels[i] = label
                    else:
                        labels[i] = f"Node {i}"
                
                nx.draw_networkx_labels(
                    graph, pos, 
                    labels=labels,
                    font_size=8,
                    font_family='DejaVu Sans'
                )
            else:
                # Add info text for large graphs
                plt.text(0.02, 0.98, 
                        f"Nodes: {len(graph.nodes())}\nEdges: {len(graph.edges())}",
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.title(f"{title}\n({len(graph.nodes())} nodes, {len(graph.edges())} edges)", 
                     fontsize=14, fontweight='bold')
            
            # Add colorbar if nodes were drawn
            if nodes is not None:
                plt.colorbar(nodes, label='Node Degree')
            
            plt.axis('off')
            plt.tight_layout()
            
            # Generate save path if not provided
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.results_dir, 
                    "visualizations", 
                    f"graph_{timestamp}.png"
                )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Graph saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error drawing graph: {e}")
            plt.close('all')
    
    def plot_interactive_graph(self, graph: nx.Graph, phrases: List[str], title: str = "Interactive Semantic Graph", save_path: Optional[str] = None):
        try:
            import plotly.graph_objects as go
            
            # Check if graph is empty
            if graph is None or len(graph.nodes()) == 0:
                logger.warning("Empty graph - cannot create interactive visualization")
                return None
            
            # Generate layout
            if len(graph.nodes()) > 100:
                pos = nx.spring_layout(graph, k=2, iterations=30, seed=42)
            else:
                pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
            
            # Prepare edge traces
            edge_x = []
            edge_y = []
            edge_text = []
            
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Get weight with default
                weight = graph[edge[0]][edge[1]].get('weight', 0.5)
                similarity = 1 - weight
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(f"Similarity: {similarity:.3f}<br>Distance: {weight:.3f}")
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='text',
                mode='lines',
                text=edge_text,
                hovertext=edge_text,
                name='Edges'
            )
            
            # Prepare node data
            node_x = []
            node_y = []
            node_degrees = []
            node_text = []
            node_labels = []
            
            for node in graph.nodes():
                node_x.append(pos[node][0])
                node_y.append(pos[node][1])
                degree = graph.degree(node)
                node_degrees.append(degree)
                
                # Create detailed hover text
                if node < len(phrases):
                    label = phrases[node]
                    # Get neighbors for additional info
                    neighbors = list(graph.neighbors(node))[:3]
                    neighbor_names = []
                    for n in neighbors:
                        if n < len(phrases):
                            neighbor_names.append(phrases[n][:15])
                        else:
                            neighbor_names.append(f"Node {n}")
                    
                    hover_text = f"<b>{label}</b><br>"
                    hover_text += f"Node ID: {node}<br>"
                    hover_text += f"Degree: {degree}<br>"
                    if neighbor_names:
                        hover_text += f"Neighbors: {', '.join(neighbor_names)}"
                        if len(list(graph.neighbors(node))) > 3:
                            hover_text += f" and {len(list(graph.neighbors(node))) - 3} more"
                else:
                    label = f"Node {node}"
                    hover_text = f"<b>{label}</b><br>Degree: {degree}"
                
                node_text.append(hover_text)
                
                # Truncate label for display
                if len(label) > 15:
                    label = label[:12] + "..."
                node_labels.append(label)
            
            # FIXED: Removed 'titleside' and properly configured colorbar
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_labels,
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    color=node_degrees,
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title=dict(
                            text='Node Degree',
                            side='right' 
                        ),
                        xanchor='left'
                    ),
                    line=dict(width=1, color='white')
                ),
                textfont=dict(size=10, color='black'),
                hovertext=node_text,
                name='Nodes'
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(
                                text=f"{title}<br><sub>{len(graph.nodes())} nodes, {len(graph.edges())} edges</sub>",
                                font=dict(size=16)
                            ),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=50),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        ))
            
            # Generate save path if not provided
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.results_dir, 
                    "visualizations", 
                    f"interactive_graph_{timestamp}.html"
                )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            fig.write_html(save_path)
            logger.info(f"Interactive graph saved to {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("Plotly is not available. Install with: pip install plotly")
            return None
        except Exception as e:
            logger.error(f"Error creating interactive graph: {e}")
            return None
    def plot_search_path(self, graph: nx.Graph, phrases: List[str], 
                        path: List[int], title: str = "Found Path",
                        save_path: Optional[str] = None):
        """
        Highlight a specific path in the graph
        
        Args:
            graph: NetworkX graph object
            phrases: List of node labels
            path: List of node indices representing the path
            title: Plot title
            save_path: Optional custom save path
        """
        try:
            # Check inputs
            if graph is None or len(graph.nodes()) == 0:
                logger.warning("Empty graph - cannot visualize path")
                return
            
            if not path or len(path) < 2:
                logger.warning("Path too short to visualize")
                return
            
            # Validate path indices
            valid_path = []
            for node in path:
                if node in graph.nodes():
                    valid_path.append(node)
                else:
                    logger.warning(f"Node {node} not in graph, skipping")
            
            if len(valid_path) < 2:
                logger.warning("No valid path nodes found")
                return
            
            plt.figure(figsize=(12, 10))
            
            # Generate layout
            if len(graph.nodes()) > 100:
                pos = nx.spring_layout(graph, k=2, iterations=30, seed=42)
            else:
                pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
            
            # Draw all nodes with low opacity
            nx.draw_networkx_nodes(
                graph, pos,
                node_color='lightgray',
                node_size=100,
                alpha=0.3
            )
            
            # Draw all edges with low opacity
            if graph.edges():
                nx.draw_networkx_edges(
                    graph, pos,
                    width=0.5,
                    alpha=0.2,
                    edge_color='gray',
                    style='dashed'
                )
            
            # Highlight path nodes
            path_subgraph = graph.subgraph(valid_path)
            nx.draw_networkx_nodes(
                path_subgraph, pos,
                node_color='red',
                node_size=300,
                alpha=0.9
            )
            
            # Draw path edges
            path_edges = []
            for i in range(len(valid_path)-1):
                if graph.has_edge(valid_path[i], valid_path[i+1]):
                    path_edges.append((valid_path[i], valid_path[i+1]))
            
            if path_edges:
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=path_edges,
                    width=3,
                    alpha=0.8,
                    edge_color='red'
                )
            
            # Add labels for path nodes
            path_labels = {}
            for i, node in enumerate(valid_path):
                if node < len(phrases):
                    label = phrases[node]
                    if len(label) > 20:
                        label = label[:17] + "..."
                    path_labels[node] = f"{i+1}: {label}"
                else:
                    path_labels[node] = f"{i+1}: Node {node}"
            
            nx.draw_networkx_labels(
                graph, pos,
                labels=path_labels,
                font_size=10,
                font_weight='bold',
                font_family='DejaVu Sans'
            )
            
            # Add path information
            path_length = len(valid_path)
            total_distance = 0
            for u, v in path_edges:
                total_distance += graph[u][v].get('weight', 0)
            
            start_phrase = phrases[valid_path[0]] if valid_path[0] < len(phrases) else f"Node {valid_path[0]}"
            end_phrase = phrases[valid_path[-1]] if valid_path[-1] < len(phrases) else f"Node {valid_path[-1]}"
            
            info_text = f"Path from '{start_phrase[:30]}' to '{end_phrase[:30]}'\n"
            info_text += f"Path length: {path_length} nodes\n"
            info_text += f"Total distance: {total_distance:.4f}"
            
            plt.text(0.02, 0.98, info_text,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.title(f"{title}", fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Generate save path
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Clean filenames
                start_str = phrases[valid_path[0]][:20].replace(' ', '_').replace('/', '_') if valid_path[0] < len(phrases) else f"node_{valid_path[0]}"
                end_str = phrases[valid_path[-1]][:20].replace(' ', '_').replace('/', '_') if valid_path[-1] < len(phrases) else f"node_{valid_path[-1]}"
                save_path = os.path.join(
                    self.results_dir, 
                    "visualizations", 
                    f"path_{start_str}_to_{end_str}_{timestamp}.png"
                )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Path visualization saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error drawing path: {e}")
            plt.close('all')
    
    def plot_algorithm_comparison(self, metrics: Dict, save_path: Optional[str] = None):
        """
        Create algorithm comparison visualization
        
        Args:
            metrics: Dictionary of PerformanceMetrics objects
            save_path: Optional custom save path
        """
        try:
            import pandas as pd
            
            # Check if metrics is empty
            if not metrics:
                logger.warning("No metrics to plot")
                return
            
            # Convert to DataFrame
            data = []
            for algo_name, metric in metrics.items():
                # Format algorithm name
                algo_display = algo_name.upper()
                
                # Convert success rate to percentage
                success_rate_pct = metric.success_rate * 100 if hasattr(metric, 'success_rate') else 0
                
                data.append({
                    'Algorithm': algo_display,
                    'Average Time (s)': metric.avg_time,
                    'Average Distance': metric.avg_distance,
                    'Visited Nodes': metric.avg_nodes_visited,
                    'Success Rate (%)': success_rate_pct,
                    'Speedup vs Dijkstra': metric.speedup_vs_dijkstra,
                    'Path Length': metric.avg_path_length
                })
            
            df = pd.DataFrame(data)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Execution Time
            ax1 = axes[0, 0]
            bars1 = ax1.bar(df['Algorithm'], df['Average Time (s)'], 
                           color='steelblue', alpha=0.8)
            ax1.set_title('Algorithm Execution Time', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars1, df['Average Time (s)']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['Average Time (s)'])*0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)
            
            # 2. Path Distance
            ax2 = axes[0, 1]
            bars2 = ax2.bar(df['Algorithm'], df['Average Distance'], 
                           color='seagreen', alpha=0.8)
            ax2.set_title('Found Path Distance', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Distance')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            for bar, val in zip(bars2, df['Average Distance']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['Average Distance'])*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
            
            # 3. Efficiency (Visited Nodes)
            ax3 = axes[1, 0]
            bars3 = ax3.bar(df['Algorithm'], df['Visited Nodes'], 
                           color='darkorange', alpha=0.8)
            ax3.set_title('Algorithm Efficiency', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Number of Visited Nodes')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            for bar, val in zip(bars3, df['Visited Nodes']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['Visited Nodes'])*0.01,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8, rotation=45)
            
            # 4. Speedup vs Dijkstra
            ax4 = axes[1, 1]
            
            # Color bars based on performance
            colors = []
            for val in df['Speedup vs Dijkstra']:
                if val > 1.1:
                    colors.append('forestgreen')
                elif val > 0.9:
                    colors.append('gold')
                else:
                    colors.append('crimson')
            
            bars4 = ax4.bar(df['Algorithm'], df['Speedup vs Dijkstra'], 
                           color=colors, alpha=0.8)
            ax4.set_title('Relative Speed vs Dijkstra', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Speedup Factor')
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Dijkstra baseline')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            for bar, val in zip(bars4, df['Speedup vs Dijkstra']):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['Speedup vs Dijkstra'])*0.02,
                        f'{val:.2f}x', ha='center', va='bottom', fontsize=8, rotation=45)
            
            plt.tight_layout()
            
            # Generate save path
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.results_dir, 
                    "visualizations", 
                    f"algorithm_comparison_{timestamp}.png"
                )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Algorithm comparison chart saved to {save_path}")
            
            # Also save data as CSV
            csv_path = save_path.replace('.png', '.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Comparison data saved to {csv_path}")
            
        except ImportError:
            logger.warning("Pandas not available for data processing")
        except Exception as e:
            logger.error(f"Error drawing comparison chart: {e}")
            plt.close('all')