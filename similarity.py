
import argparse
import yaml
from typing import Dict, Any, Optional
import networkx as nx

from src.model.models import MERGERS
from src.model.graph_merger import GraphAbstractionMerger
from src.model.singleton import handle_config_singleton


def parse_args():
    parser = argparse.ArgumentParser(description='Build a graph from a csv file')
    parser.add_argument('config', type=str, help='Path to the data configuration file')
    return parser.parse_args()



def main(graph_path_1 : str,  
         graph_path_2 : str,
         merger_class : Optional[str] = None, 
         merger_config : Optional[Dict[str,Any]] = None):
    
    # Load graphs
    graph_1 = nx.read_gml(graph_path_1)
    graph_2 = nx.read_gml(graph_path_2)

    # Rename nodes
    nx.relabel_nodes(graph_1, {node: f"g1-{node}" for node in graph_1.nodes}, copy=False)
    nx.relabel_nodes(graph_2, {node: f"g2-{node}" for node in graph_2.nodes}, copy=False)

    # Build merger
    print(f"Using merger {merger_class}.")
    merger = MERGERS[merger_class](**merger_config)

    if not isinstance(merger, GraphAbstractionMerger):
        raise ValueError("Merger must be a GraphAbstractionMerger")
    
    # Find similar nodes
    similar_nodes = merger._find_similar_nodes([graph_1, graph_2])

    # Update node names to make the two graphs comparable    
    nx.relabel_nodes(graph_1, {old_name: new_name for old_name, (new_name, _) in similar_nodes[0].items()}, copy=False)
    nx.relabel_nodes(graph_2, {old_name: new_name for old_name, (new_name, _) in similar_nodes[1].items()}, copy=False)

    # Compute graph edit distance similarity
    ged = nx.graph_edit_distance(graph_1, graph_2)
    print(f"Graph Edit Distance: {ged}")

    # Compute intersection over union similarity
    intersection = nx.intersection(graph_1, graph_2)
    union = nx.compose(graph_1, graph_2)
    iou_ged = nx.graph_edit_distance(intersection, union)
    print(f"Intersection over Union Graph Edit Distance: {iou_ged}")



if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = handle_config_singleton(config)
    
    main(**config)