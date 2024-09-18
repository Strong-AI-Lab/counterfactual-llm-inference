
import os
import yaml
import time
import argparse
from typing import List, Union, Dict, Any, Union, Optional
import tqdm
import networkx as nx

from src.data.dataset import DATASETS
from src.model.graph_builder import BUILDERS
from src.model.graph_merger import MERGERS
from src.visualisation.visualisation import save_graph_as_png



DEFAULT_GRAPH_SAVE_FOLDER = 'causal_graphs'


def parse_args():
    parser = argparse.ArgumentParser(description='Build a graph from a csv file')
    parser.add_argument('config', type=str, help='Path to the data configuration file')
    return parser.parse_args()




def main(data_path : Union[str,List[str]], 
         dataset_class : str, 
         dataset_config : Dict[str,Any],
         builder_class : str, 
         builder_config : Dict[str,Any], 
         merger_class : Optional[str] = None, 
         merger_config : Optional[Dict[str,Any]] = None, 
         output_path : Optional[str] = None):
    
    # Load text data
    print(f"Using dataset {dataset_class}.")
    data = DATASETS[dataset_class](data_path, **dataset_config)

    # Load graph builder
    print(f"Using builder {builder_class}.")
    builder = BUILDERS[builder_class](**builder_config)

    # Build graph
    graphs = []
    for name, text in tqdm.tqdm(data):
        graph = builder.build_graph(name, text)
        graphs.append(graph)

    # Merge graphs
    if len(graphs) > 1:
        print(f"Merging {len(graphs)} graphs with merger {merger_class}.")
        merger = MERGERS[merger_class](**merger_config)
        merged_graph = merger.merge_graphs(graphs)
    else:
        merged_graph = graphs[0]
    
    # Save graph
    if output_path is None:
        output_path = os.path.join(DEFAULT_GRAPH_SAVE_FOLDER, f'graph_{time.strftime("%Y%m%d-%H%M%S")}.gml')
    print(f"Saving graph to {output_path}.")

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