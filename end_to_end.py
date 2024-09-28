
import os
import yaml
import time
import argparse
from typing import List, Union, Dict, Any, Union, Optional, Callable
import tqdm
import random
import networkx as nx

from src.data.dataset import DATASETS
from src.model.models import BUILDERS, MERGERS, NODE_INTERPRETERS, INFERENCE_ORACLES, EVALUATORS
from src.causal.counterfactual_inference import Query
from src.model.singleton import handle_config_singleton
from src.visualisation.visualisation import save_graph_as_png


DEFAULT_SCORED_GRAPHS_SAVE_FOLDER = 'scored_graphs'



def parse_args():
    parser = argparse.ArgumentParser(description='Compute counterfactual query on causal graph')
    parser.add_argument('config', type=str, help='Path to the query configuration file')
    return parser.parse_args()


def generate_random_counterfactual_query(graph : nx.DiGraph, max_intervention_nodes : int, interpreter : Callable) -> Dict[str,Any]:
    candidate_targets = [node for node in graph.nodes if len(list(graph.predecessors(node))) > 0]
    target_node = random.choice(list(candidate_targets))

    candidate_interventions = list(nx.ancestors(graph, target_node))
    nb_interventions = random.randint(1, min(max_intervention_nodes, len(candidate_interventions)))
    intervention_nodes = random.sample(candidate_interventions, nb_interventions)

    intervention_values = [interpreter(graph.nodes[intervention_node]) for intervention_node in intervention_nodes]

    return {'target_node': target_node,
            'observation_nodes': None, # no observations
            'intervention_nodes': intervention_nodes,
            'observation_values': None, # no observations
            'intervention_values': intervention_values,}


def main(data_path : Union[str,List[str]], 
         dataset_class : str, 
         dataset_config : Dict[str,Any],
         evaluator_class : str, 
         evaluator_config : Dict[str,Any], 
         oracle_class : str, 
         oracle_config : Dict[str,Any], 
         interpreter_class : str, 
         interpreter_config : Dict[str,Any], 
         builder_class : str, 
         builder_config : Dict[str,Any], 
         merger_class : Optional[str] = None, 
         merger_config : Optional[Dict[str,Any]] = None, 
         iterations : int = 1,
         num_graphs : int = 2,
         num_queries : int = 5,
         max_intervention_nodes : int = 3,
         graph_traversal_cutoff : Optional[int] = None,
         output_path : Optional[str] = None):
    
    # Load text data
    print(f"Using dataset {dataset_class}.")
    data = DATASETS[dataset_class](data_path, **dataset_config)

    # Load graph builder: build causal graphs from text data
    print(f"Using builder {builder_class}.")
    builder = BUILDERS[builder_class](**builder_config)

    # Load graph merger: merge multiple causal graphs into one
    if len(data) > 1:
        print(f"Using merger {merger_class}.")
        merger = MERGERS[merger_class](**merger_config)

    # Load interpreter: interpret a node to build alternative/counterfactual instantiations
    print(f"Using interpreter {interpreter_class}.")
    interpreter = NODE_INTERPRETERS[interpreter_class](**interpreter_config)

    # Load oracle: answer counterfactual queries
    print(f"Using oracle {oracle_class}.")
    oracle = INFERENCE_ORACLES[oracle_class](**oracle_config)

    # Load evaluator: evaluate the plausibility of counterfactual graphs
    print(f"Using evaluator {evaluator_class}.")
    evaluator = EVALUATORS[evaluator_class](**evaluator_config)


    if output_path is None:
        output_path = os.path.join(DEFAULT_SCORED_GRAPHS_SAVE_FOLDER, f'scored_graphs_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(output_path, exist_ok=True)


    # Iteratively build causal graphs
    for i in tqdm.trange(iterations):
        # Build graphs
        graphs = []
        for j in tqdm.trange(num_graphs, leave=False):
            data_graphs = []
            for name, text in tqdm.tqdm(data, leave=False):
                graph = builder.build_graph(name, text)
                data_graphs.append(graph)

            if len(data_graphs) > 1:
                merged_data_graph = merger.merge_graphs(data_graphs)
            else:
                merged_data_graph = data_graphs[0]
            
            graphs.append(merged_data_graph)

        # Answer counterfactual queries (num_queries per graph)
        counterfactual_graphs = []
        for graph in tqdm.tqdm(graphs, leave=False):
            cg_j = [graph]
            for _ in tqdm.trange(num_queries, leave=False):
                query_config = generate_random_counterfactual_query(graph, max_intervention_nodes, interpreter)

                # Compute counterfactuals
                query = Query(graph, oracle, **query_config, traversal_cutoff=graph_traversal_cutoff, compute_counterfactuals=True)
                print(repr(query))

                _, computation_graph = query()
                cg_j.append(computation_graph)
            counterfactual_graphs.append(cg_j)


        # Evaluate counterfactual graphs
        scores = []
        for cg_j in tqdm.tqdm(counterfactual_graphs):
            scores_cgj = []
            for g in cg_j:
                score, confidence, explanation = evaluator.evaluate(g)
                scores_cgj.append(score)
                g.graph['plausibility_score'] = score
                g.graph['plausibility_score_confidence'] = confidence
                g.graph['plausibility_score_explanation'] = explanation
            scores.append(scores_cgj)
        

        # Save graphs
        output_path_i = os.path.join(output_path, f'iteration_{i}')
        os.makedirs(output_path_i, exist_ok=True)

        for j, queries_cgi in enumerate(counterfactual_graphs):
            score_cgi = sum(scores[j]) / len(scores[j])
            output_path_ij = os.path.join(output_path_i, f'graph_{j}_score={score_cgi}')
            os.makedirs(output_path_ij, exist_ok=True)

            nx.write_gml(queries_cgi[0], os.path.join(output_path_ij, f'initial_graph_score={scores[j][0]}.gml'))
            save_graph_as_png(queries_cgi[0], os.path.join(output_path_ij, f'initial_graph_score={scores[j][0]}.png'), node_labels=['description', 'type', 'values', 'current_value', 'context'], edge_labels=['description', 'details'])

            for k, g in enumerate(queries_cgi[1:]):
                nx.write_gml(g, os.path.join(output_path_ij, f'counterfactual_graph_{k}_score={scores[j][k+1]}.gml'))
                save_graph_as_png(g, os.path.join(output_path_ij, f'counterfactual_graph_{k}_score={scores[j][k+1]}.png'), node_labels=['description','current_value','updated_value'])
        

        # Update graph builder and oracle
        # builder.update(scores)
        # oracle.update(scores)
                




if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = handle_config_singleton(config)
    main(**config)