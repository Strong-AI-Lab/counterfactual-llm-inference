
import argparse
import os
import time
import yaml
from typing import Any, Dict, Optional, Union, List
import networkx as nx

from src.causal.counterfactual_inference import Query
from src.model.models import INFERENCE_ORACLES
from src.visualisation.visualisation import save_graph_as_png



DEFAULT_COUNTERFACTUALS_SAVE_FOLDER = 'counterfactuals'


def parse_args():
    parser = argparse.ArgumentParser(description='Compute counterfactual query on causal graph')
    parser.add_argument('config', type=str, help='Path to the query configuration file')
    return parser.parse_args()



def main(graph_save : str, 
         oracle_class : str, 
         oracle_config : Dict[str,Any], 
         target_node : str, 
         observation_nodes : Optional[Union[str, List[str]]] = None, 
         intervention_nodes : Optional[Union[str, List[str]]] = None, 
         observation_values : Optional[Union[str, List[str]]] = None, 
         intervention_values : Optional[Union[str, List[str]]] = None,
         is_counterfactual : Optional[bool] = False,
         graph_traversal_cutoff : Optional[int] = None,
         output_path : Optional[str] = None):
    
    # Load graph
    print(f"Loading graph from {graph_save}.")
    graph = nx.read_gml(graph_save)

    # Load oracle
    print(f"Using oracle {oracle_class}.")
    oracle = INFERENCE_ORACLES[oracle_class](**oracle_config)

    # Compute counterfactuals
    query = Query(graph, oracle, target_node, observation_nodes, intervention_nodes, observation_values, intervention_values, is_counterfactual, graph_traversal_cutoff)
    print(repr(query))

    answer, inference_graph = query()

    # Print answer
    print(f"Value of target node {target_node} in initial graph is:\n----------\n{graph.nodes[target_node]['current_value']}\n----------")
    print(f"New value of target node {target_node} is:\n----------\n{answer}\n----------")

    # Save inference graph
    if output_path is None:
        output_path = os.path.join(DEFAULT_COUNTERFACTUALS_SAVE_FOLDER, f'counterfactual_graph_{time.strftime("%Y%m%d-%H%M%S")}.gml')
    print(f"Saving inference graph to {output_path}.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_gml(inference_graph, output_path)

    # Save graph as png
    png_output_path = output_path.replace('.gml', '.png')
    save_graph_as_png(inference_graph, png_output_path, node_labels='updated_value')



if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(**config)