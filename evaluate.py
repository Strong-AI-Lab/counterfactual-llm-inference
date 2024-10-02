
import os
import yaml
import time
import argparse
import random
from typing import List, Union, Dict, Any, Union, Optional, Callable, Tuple
import tqdm
import networkx as nx

from src.data.evaluation_dataset import EVALUATION_DATASETS 
from src.model.models import BUILDERS, MERGERS, QUERY_INTERPRETERS, INFERENCE_ORACLES, EVALUATORS
from src.causal.counterfactual_inference import Query
from src.model.singleton import handle_config_singleton
from src.visualisation.visualisation import save_graph_as_png
from src.utils.evaluation_utils import sem_equals

from json.decoder import JSONDecodeError
from langchain_core.exceptions import OutputParserException
from networkx.exception import NetworkXError, NetworkXUnfeasible


DEFAULT_EVALUATION_GRAPHS_SAVE_FOLDER = 'evaluation_logs'



def parse_args():
    parser = argparse.ArgumentParser(description='Compute end-to-end causal discovery and inference on dataset with ground truth')
    parser.add_argument('config', type=str, help='Path to the query configuration file')
    return parser.parse_args()


def build_query(graph : nx.DiGraph, query_text : str, interpreter : Callable) -> Dict[str,Any]:
    nodes = list(graph.nodes(data=True))
    query = interpreter(query_text, nodes)

    return {'target_node': query['target_variable'],
            'observation_nodes': None, # no observations
            'intervention_nodes': query['intervention_variable'],
            'observation_values': None, # no observations
            'intervention_values': query['intervention_new_value'],
            }

def build_topological_graph(graph : nx.DiGraph) -> nx.DiGraph:
    order = list(nx.topological_sort(graph))
    node_relabel = {node: str(i) for i, node in enumerate(order)}
    return nx.relabel_nodes(graph, node_relabel, copy=True)


def match_attributes(graph : nx.DiGraph, ref_graph : nx.DiGraph):
    graph = graph.copy()

    node_relabel = {}
    for node in graph.nodes:
        graph.nodes[node]['observed'] = True
        for ref_node in ref_graph.nodes:
            if node.lower() == ref_graph.nodes[ref_node]['description'].lower():
                graph.nodes[node]['description'] = ref_graph.nodes[ref_node]['description']
                node_relabel[node] = ref_node
                break
    
    nx.relabel_nodes(graph, node_relabel, copy=False)
    
    return graph

def save_errors(output_path : str, iteration_name : str, error : str):
    output_path_i = os.path.join(output_path, f'iteration_{iteration_name}')
    os.makedirs(output_path_i, exist_ok=True)

    with open(os.path.join(output_path_i, 'errors.txt'), 'w') as f:
        f.write(error)


def process_item(item : Dict[str,Any], 
                 builder : Callable, 
                 oracle : Callable, 
                 interpreter : Callable, 
                 evaluator : Callable, 
                 graph_traversal_cutoff : Optional[int] = None,
                 no_build_graph : bool = False
                 ) -> Tuple[Dict[str,Any], nx.DiGraph, nx.DiGraph, nx.DiGraph]:
        
    name = item['name']
    text = item['text']
    query_text = item['query']
    gt_graph = item['gt_graph']
    label = item['label']
    
    # Build graph
    if no_build_graph:
        graph = gt_graph
    else:
        try:
            graph = builder.build_graph(name, text)
        except (KeyError, JSONDecodeError, OutputParserException, NetworkXError) as e:
            raise e.__class__(f"Error building graph: {e}")

    # Answer counterfactual queries (num_queries per graph)
    query_config = build_query(graph, query_text, interpreter)

    # Compute counterfactuals
    try:
        query = Query(graph, oracle, **query_config, traversal_cutoff=graph_traversal_cutoff, compute_counterfactuals=True)
        answer, computation_graph = query()
    except (KeyError, JSONDecodeError, OutputParserException) as e:
        raise e.__class__(f"Error computing counterfactuals: {e}")

    # Evaluate counterfactual graphs
    score = {
        'item' : item,
    }

    # Compute label accuracy
    correct = sem_equals(answer, label)
    score['correct'] = correct
    score['answer'] = answer
    score['label'] = label
    score['question_property'] = item['question_property']

    # Update ground truth graph
    gt_graph = match_attributes(gt_graph, graph)

    # Compute graph self-scores
    # for g, name in [(gt_graph, "ground_truth"), (graph, "estimated"), (computation_graph, "counterfactual")]:
    for g, g_name in [(graph, "estimated"), (computation_graph, "counterfactual")]:
        try:
            plausibility, confidence, explanation = evaluator.evaluate(g)
        except (JSONDecodeError, OutputParserException, NetworkXUnfeasible) as e:
            plausibility, confidence, explanation = None, None, None

        g.graph['plausibility_score'] = plausibility
        g.graph['plausibility_score_confidence'] = confidence
        g.graph['plausibility_score_explanation'] = explanation

        score[f'{g_name}_plausibility'] = plausibility
        score[f'{g_name}_confidence'] = confidence
        score[f'{g_name}_explanation'] = explanation

    # Compute graph edit distance (ged) similarity
    max_nodes = max(gt_graph.number_of_nodes(), graph.number_of_nodes())
    ged = nx.graph_edit_distance(gt_graph, graph)
    norm_ged = ged / max_nodes
    score['graph_edit_distance'] = ged
    score['normalized_graph_edit_distance'] = norm_ged

    # Compute intersection over union similarity
    intersection = nx.intersection(gt_graph, graph)
    union = nx.compose(gt_graph, graph)
    iou_ged = nx.graph_edit_distance(intersection, union)
    norm_iou_ged = iou_ged / max_nodes
    score['intersection_over_union_graph_edit_distance'] = iou_ged
    score['normalized_intersection_over_union_graph_edit_distance'] = norm_iou_ged

    # Compute label-free ged
    if nx.is_directed_acyclic_graph(gt_graph) and nx.is_directed_acyclic_graph(graph):
        gt_graph_topo = build_topological_graph(gt_graph)
        graph_topo = build_topological_graph(graph)
        ged_topo = nx.graph_edit_distance(gt_graph_topo, graph_topo)
        norm_ged_topo = ged_topo / max_nodes

        # Compute label-free iou_ged
        intersection_topo = nx.intersection(gt_graph_topo, graph_topo)
        union_topo = nx.compose(gt_graph_topo, graph_topo)
        iou_ged_topo = nx.graph_edit_distance(intersection_topo, union_topo)
        norm_iou_ged_topo = iou_ged_topo / max_nodes
    else:
        ged_topo = None
        norm_ged_topo = None
        iou_ged_topo = None
        norm_iou_ged_topo = None
    
    score['graph_edit_distance_topological'] = ged_topo
    score['normalized_graph_edit_distance_topological'] = norm_ged_topo
    score['intersection_over_union_graph_edit_distance_topological'] = iou_ged_topo
    score['normalized_intersection_over_union_graph_edit_distance_topological'] = norm_iou_ged_topo

    return score, gt_graph, graph, computation_graph



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
         graph_traversal_cutoff : Optional[int] = None,
         no_build_graph : bool = False,
         save_graphs : bool = False,
         save_logs : bool = True,
         nb_queries : Optional[int] = None, # for debugging purposes
         output_path : Optional[str] = None):
    
    # Load text data
    print(f"Using dataset {dataset_class}.")
    data = EVALUATION_DATASETS[dataset_class](data_path, **dataset_config)
    
    if nb_queries is not None:
        random.seed(42)
        indices = random.sample(range(len(data)), nb_queries)
        data = data[indices]

    # Load graph builder: build causal graphs from text data
    print(f"Using builder {builder_class}.")
    builder = BUILDERS[builder_class](**builder_config)

    # Load oracle: answer counterfactual queries
    print(f"Using oracle {oracle_class}.")
    oracle = INFERENCE_ORACLES[oracle_class](**oracle_config)

    # Load interpreter: interpret a node to build alternative/counterfactual instantiations
    print(f"Using interpreter {interpreter_class}.")
    interpreter = QUERY_INTERPRETERS[interpreter_class](**interpreter_config)

    # Load evaluator: evaluate the plausibility of counterfactual graphs
    print(f"Using evaluator {evaluator_class}.")
    evaluator = EVALUATORS[evaluator_class](**evaluator_config)


    if output_path is None:
        evaluated_model = ''
        if 'model_type' in builder_config:
            evaluated_model += f"{builder_config['model_type']}_"
        if 'model' in builder_config:
            evaluated_model += f"{builder_config['model']}_"
        output_path = os.path.join(DEFAULT_EVALUATION_GRAPHS_SAVE_FOLDER, f'logs_evaluation_{dataset_class}_{evaluated_model}_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(output_path, exist_ok=True)


    # Iteratively build causal graphs
    scores = []
    for item in tqdm.tqdm(data):
        try:
            score, gt_graph, graph, computation_graph = process_item(item, builder, oracle, interpreter, evaluator, graph_traversal_cutoff, no_build_graph)
            scores.append(score)

            if save_logs or save_graphs:
                output_path_i = os.path.join(output_path, f"iteration_{item['name']}")
                os.makedirs(output_path_i, exist_ok=True)
            
            if save_logs:
                # Write logs
                with open(os.path.join(output_path_i, 'logs.txt'), 'w') as f:
                    for key, value in score.items():
                        f.write(f"{key}: {value}\n")

            if save_graphs:
                # Write graph
                nx.write_gml(gt_graph, os.path.join(output_path_i, f'ground_truth_graph.gml'))
                save_graph_as_png(gt_graph, os.path.join(output_path_i, f'ground_truth_graph.png'), node_labels=[], edge_labels=[])

                nx.write_gml(graph, os.path.join(output_path_i, f'initial_graph_score={score["estimated_plausibility"]}.gml'))
                save_graph_as_png(graph, os.path.join(output_path_i, f'initial_graph_score={score["estimated_plausibility"]}.png'), node_labels=['description', 'type', 'values', 'current_value', 'context'], edge_labels=['description', 'details'])

                nx.write_gml(computation_graph, os.path.join(output_path_i, f'counterfactual_graph_score={score["counterfactual_plausibility"]}.gml'))
                save_graph_as_png(computation_graph, os.path.join(output_path_i, f'counterfactual_graph_score={score["counterfactual_plausibility"]}.png'), node_labels=['description','current_value','updated_value'])

        except Exception as e:
            save_errors(output_path, item['name'], str(e))





if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = handle_config_singleton(config)
    main(**config)