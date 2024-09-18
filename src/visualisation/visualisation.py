
from typing import List
import networkx as nx
import matplotlib.pyplot as plt


def attrs_to_text(attrs : dict, labels : List[str]) -> str:
    return ', '.join([f"{label}: {attrs[label]}" for label in labels if label in attrs])


def save_graph_as_png(graph : nx.Graph, output_path : str, node_labels : str | List[str] = 'description', edge_labels : str | List[str] = 'description'):
    graph = graph.copy()
    if nx.is_directed_acyclic_graph(graph): # create topological order
        partial_order = list(nx.topological_generations(graph))
        for layer, nodes in enumerate(partial_order[::-1]):
            for node in nodes:
                graph.nodes[node]['layer'] = layer
    
    # Plot graph and event legend into 3 subplots
    fig, ax = plt.subplots(1,3,figsize=(34,12))

    # Plot graph
    if nx.is_directed_acyclic_graph(graph):
        pos = nx.multipartite_layout(graph, subset_key="layer", align='horizontal')
    else:
        pos = nx.spring_layout(graph, k=1/(len(graph.nodes)**(1/3)))
    nx.draw(graph, pos, ax=ax[0], node_size=600, width=0.5, with_labels=True, font_size=10, font_color='black', node_color=[('skyblue' if attrs['observed'] else 'lightcoral') for node, attrs in graph.nodes(data=True)], edge_color='gray')

    if isinstance(node_labels, str):
        node_labels = [node_labels]
    if isinstance(edge_labels, str):
        edge_labels = [edge_labels]

    # Plot node legend
    ax[1].axis('off')
    ax[1].text(0, 0.9, "Causal Variables", fontsize=12, fontweight='bold')
    for i, (j, attrs) in enumerate(graph.nodes(data=True)):
        ax[1].text(0, 0.85 - 0.04*int(i), f"{j}: {attrs_to_text(attrs, node_labels)}", fontsize=10)

    # Plot relation legend
    ax[2].axis('off')
    ax[2].text(0, 0.9, "Causal Relationships", fontsize=12, fontweight='bold')
    for i, (j, k, attrs) in enumerate(graph.edges(data=True)):
        ax[2].text(0, 0.85 - 0.04*int(i), f"{j} -> {k}: {attrs_to_text(attrs, edge_labels)}", fontsize=10)

    # Save
    fig.savefig(output_path)