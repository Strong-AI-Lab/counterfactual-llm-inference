
import os
import yaml
import time
import argparse
from typing import List, Union, Dict, Any, Union, Optional
import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from src.data.dataset import DATASETS
from src.model.graph_builder import BUILDERS
from src.model.graph_merger import MERGERS



DEFAULT_GRAPH_SAVE_FOLDER = 'causal_graphs'


def parse_args():
    parser = argparse.ArgumentParser(description='Build a graph from a csv file')
    parser.add_argument('config', type=str, help='Path to the data configuration file')
    return parser.parse_args()


def save_graph_as_png(graph : nx.Graph, output_path : str):
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

    # Plot node legend
    ax[1].axis('off')
    ax[1].text(0, 0.9, "Causal Variables", fontsize=12, fontweight='bold')
    for i, (j, attrs) in enumerate(graph.nodes(data=True)):
        ax[1].text(0, 0.85 - 0.04*int(i), f"{j}: {attrs['description']}", fontsize=10)

    # Plot relation legend
    ax[2].axis('off')
    ax[2].text(0, 0.9, "Causal Relationships", fontsize=12, fontweight='bold')
    for i, (j, k, attrs) in enumerate(graph.edges(data=True)):
        ax[2].text(0, 0.85 - 0.04*int(i), f"{j} -> {k}: {attrs['description']}", fontsize=10)

    # Save
    fig.savefig(output_path)



def main(data_path : Union[str,List[str]], dataset_class : str, dataset_config : Dict[str,Any], builder_class : str, builder_config : Dict[str,Any], merger_class : Optional[str] = None, merger_config : Optional[Dict[str,Any]] = None, output_path : Optional[str] = None):
    # Load text data
    data = DATASETS[dataset_class](data_path, **dataset_config)

    # Load graph builder
    builder = BUILDERS[builder_class](**builder_config)

    # Build graph
    graphs = []
    for name, text in tqdm.tqdm(data):
        graph = builder.build_graph(name, text)
        graphs.append(graph)

    if len(graphs) > 1:
        merger = MERGERS[merger_class](**merger_config)
        merged_graph = merger.merge_graphs(graphs)
    else:
        merged_graph = graphs[0]
    
    # Save graph
    if output_path is None:
        output_path = os.path.join(DEFAULT_GRAPH_SAVE_FOLDER, f'graph_{time.strftime("%Y%m%d-%H%M%S")}.gml')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_gml(merged_graph, output_path)

    # Save graph as png
    png_output_path = output_path.replace('.gml', '.png')
    save_graph_as_png(merged_graph, png_output_path)



if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(**config)