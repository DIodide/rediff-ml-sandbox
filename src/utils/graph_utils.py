"""
Utility functions for graph machine learning operations.
"""

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from typing import Optional, Dict, Any


def networkx_to_pyg(
    G: nx.Graph, node_attr: Optional[str] = None, target_attr: Optional[str] = None
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Args:
        G: NetworkX graph
        node_attr: Name of node attribute to use as features
        target_attr: Name of node attribute to use as targets

    Returns:
        PyTorch Geometric Data object
    """
    # Get edge list
    edge_list = list(G.edges())

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Node features
    if node_attr and node_attr in G.nodes[list(G.nodes())[0]]:
        x = torch.tensor(
            [G.nodes[node][node_attr] for node in G.nodes()], dtype=torch.float
        )
        if x.dim() == 1:
            x = x.unsqueeze(1)
    else:
        # Random features if no attributes provided
        x = torch.randn(G.number_of_nodes(), 16)

    # Targets
    y = None
    if target_attr and target_attr in G.nodes[list(G.nodes())[0]]:
        y = torch.tensor(
            [G.nodes[node][target_attr] for node in G.nodes()], dtype=torch.long
        )

    return Data(x=x, edge_index=edge_index, y=y)


def pyg_to_networkx(data: Data) -> nx.Graph:
    """
    Convert PyTorch Geometric Data object to NetworkX graph.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        NetworkX graph
    """
    G = nx.Graph()

    # Add nodes
    num_nodes = (
        data.x.size(0) if data.x is not None else data.edge_index.max().item() + 1
    )
    G.add_nodes_from(range(num_nodes))

    # Add edges (remove duplicates for undirected graph)
    edge_list = data.edge_index.t().tolist()
    edges_set = set()
    for edge in edge_list:
        edge_tuple = tuple(sorted(edge))
        edges_set.add(edge_tuple)

    G.add_edges_from(list(edges_set))

    return G


def compute_graph_stats(G: nx.Graph) -> Dict[str, Any]:
    """
    Compute basic statistics for a graph.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of graph statistics
    """
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G),
    }

    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        stats.update(
            {
                "avg_degree": sum(degrees.values()) / len(degrees),
                "max_degree": max(degrees.values()),
                "min_degree": min(degrees.values()),
            }
        )

        if nx.is_connected(G):
            stats.update(
                {
                    "diameter": nx.diameter(G),
                    "avg_path_length": nx.average_shortest_path_length(G),
                    "avg_clustering": nx.average_clustering(G),
                }
            )
        else:
            # For disconnected graphs, compute for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            stats.update(
                {
                    "num_components": nx.number_connected_components(G),
                    "largest_component_size": len(largest_cc),
                    "largest_component_diameter": nx.diameter(subgraph),
                    "avg_clustering": nx.average_clustering(G),
                }
            )

    return stats


def create_random_graph(
    num_nodes: int, edge_prob: float = 0.1, num_features: int = 16, num_classes: int = 2
) -> Data:
    """
    Create a random graph for testing.

    Args:
        num_nodes: Number of nodes
        edge_prob: Probability of edge between any two nodes
        num_features: Number of node features
        num_classes: Number of classes for node labels

    Returns:
        PyTorch Geometric Data object
    """
    # Create random graph
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)

    # Convert to PyG format
    data = networkx_to_pyg(G)

    # Add random features and labels
    data.x = torch.randn(num_nodes, num_features)
    data.y = torch.randint(0, num_classes, (num_nodes,))

    return data


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Add self-loops to edge index.

    Args:
        edge_index: Edge index tensor
        num_nodes: Number of nodes

    Returns:
        Edge index with self-loops
    """
    self_loops = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    return edge_index


def normalize_adjacency(
    edge_index: torch.Tensor, num_nodes: int, add_self_loops: bool = True
) -> torch.Tensor:
    """
    Normalize adjacency matrix for GCN.

    Args:
        edge_index: Edge index tensor
        num_nodes: Number of nodes
        add_self_loops: Whether to add self-loops

    Returns:
        Normalized edge weights
    """
    from torch_geometric.utils import degree

    if add_self_loops:
        edge_index = add_self_loops(edge_index, num_nodes)

    # Compute degree
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    # Compute normalized edge weights
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_weight
