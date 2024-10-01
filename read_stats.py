
import os
import argparse
import tqdm
import re
from typing import List

from src.utils.evaluation_utils import sem_equals, validate_sentence

class Stats:

    def __init__(self, keys : List[str]):
        self.keys = keys
        self.stats = {}
        self.count = {}

        for key in keys:
            self.stats[key] = 0
            self.count[key] = 0

    def add(self, key : str, value : float):
        if value is not None:
            self.stats[key] += value
            self.count[key] += 1

    def add_dict(self, stats : dict):
        for key, value in stats.items():
            self.add(key, value)

    def get(self):
        return self.stats

    def get_mean(self):
        mean = {}
        for key in self.keys:
            mean[key] = 0 if self.count[key] == 0 else self.stats[key] / self.count[key]

        return mean



LOG_REG = re.compile(r'(?P<key>\w+): (?P<value>.*)')
LOG_QP_REG = re.compile(r'\'question_property\': \'(?P<question_property>\w+)\'')
QUERY_REG = re.compile(r"('|\")query('|\"): ('|\")Would (?P<tested_estimand>.*)? if")
GRAPH_ERR_REG = re.compile(r'Error building graph:')
INFERENCE_ERR_REG = re.compile(r'Error computing counterfactuals:')
ACYCLIC_ERR_REG = re.compile(r'graph should be directed acyclic')


def fix_answer(answer : str, item : str) -> str:
    estimand = QUERY_REG.search(item).group('tested_estimand')
    val = validate_sentence(estimand, answer)
    if val:
        return 'yes'
    else:
        return 'no'
    

def to_float(value : str) -> float:
    if value is None or value == 'None':
        return None
    return float(value)

def read_sample_log(sample_log : str) -> dict:
    log_stats = {}
    with open(sample_log, 'r') as f:
        for line in f:
            match = LOG_REG.match(line)
            if match:
                log_stats[match.group('key')] = match.group('value')

    if 'question_property' not in log_stats:
        question_property = LOG_QP_REG.search(log_stats['item'])
        log_stats['question_property'] = question_property.group('question_property')
    
    unanswered = None
    if log_stats['answer'].lower() not in ['yes', 'true', '1', 'y', 't', 'no', 'false', '0', 'n', 'f']:
        try:
            fixed_answer = fix_answer(log_stats['answer'], log_stats['item'])
            fixed_correct = sem_equals(fixed_answer, log_stats['label'])
            correct = fixed_correct
        except ValueError as e:
            print(e)
            fixed_correct = None
            correct = None
            unanswered = 1
        correct_formatted = None
    else:
        fixed_correct = None
        correct_formatted = True if log_stats['correct'] == 'True' else False
        correct = correct_formatted

    sense = {
        'commonsense' : None,
        'nonsense' : None,
        'anticommonsense' : None
    }
    sense_correct = {
        'commonsense_correct' : None,
        'nonsense_correct' : None,
        'anticommonsense_correct' : None
    }
    sense[log_stats['question_property']] = 1
    sense_correct[log_stats['question_property'] + '_correct'] = correct

    return {
        'correct' : correct,
        'formatted_correct' : correct_formatted, # correct answers out of all correctly formatted answers
        'fixed_correct' : fixed_correct,
        'formatted' : True if correct_formatted is not None else False,
        'fixed' : True if fixed_correct is not None else False,
        'plausibility' : to_float(log_stats['estimated_plausibility']) if to_float(log_stats['estimated_plausibility'])  and to_float(log_stats['estimated_plausibility']) >= 0.0 else None, 
        'confidence' : to_float(log_stats['estimated_confidence']) if to_float(log_stats['estimated_confidence']) and to_float(log_stats['estimated_confidence']) >= 0.0 else None,
        'counterfactual_plausibility' : to_float(log_stats['counterfactual_plausibility']) if to_float(log_stats['counterfactual_plausibility']) and to_float(log_stats['counterfactual_plausibility']) >= 0.0 else None,
        'counterfactual_confidence' : to_float(log_stats['counterfactual_confidence']) if to_float(log_stats['counterfactual_confidence']) and to_float(log_stats['counterfactual_confidence']) >= 0.0 else None,
        'ged' : float(log_stats['graph_edit_distance']),
        'iou' : float(log_stats['intersection_over_union_graph_edit_distance']),
        'is_topo' : log_stats['graph_edit_distance_topological'] is not None,
        'topo_ged' : to_float(log_stats['graph_edit_distance_topological']),
        'topo_iou' : to_float(log_stats['intersection_over_union_graph_edit_distance_topological']),
        'unanswered' : unanswered,
        **sense,
        **sense_correct
    }


def read_errors(sample_log : str) -> dict:
    log_errs = {}
    with open(sample_log, 'r') as f:
        first_line = f.readline()
        if re.search(GRAPH_ERR_REG, first_line):
            log_errs['errors_inference'] = 1
        elif re.search(INFERENCE_ERR_REG, first_line):
            log_errs['errors_building_graph'] = 1
        elif re.search(ACYCLIC_ERR_REG, first_line):
            log_errs['errors_acyclicity'] = 1
        else:
            print('Unknown error:', first_line, first_line[0])
            log_errs['unknown_errors'] = 1

    return log_errs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folder', type=str)
    args = parser.parse_args()
    save_folder = args.save_folder

    sample_logs = os.listdir(save_folder)
    print('Number of samples:', len(sample_logs))

    stats = Stats(["completed", "errors", "correct", "formatted_correct", "fixed_correct", 
                    "formatted", "fixed", "plausibility", "confidence",
                    "counterfactual_plausibility", "counterfactual_confidence", 
                    "ged", "iou", "is_topo", "topo_ged", "topo_iou",
                    "commonsense", "nonsense", "anticommonsense", 
                    "commonsense_correct", "nonsense_correct", "anticommonsense_correct",
                    "errors_building_graph", "errors_inference", "errors_acyclicity", "unanswered", "unknown_errors"
                ])


    for sample_log in tqdm.tqdm(sample_logs):
        if os.path.exists(os.path.join(save_folder, sample_log, "logs.txt")):
            log_values = read_sample_log(os.path.join(save_folder, sample_log, "logs.txt"))
            log_values['completed'] = 1
            stats.add_dict(log_values)
        if os.path.exists(os.path.join(save_folder, sample_log, "errors.txt")):
            err_values = read_errors(os.path.join(save_folder, sample_log, "errors.txt"))
            err_values["errors"] = 1
            stats.add_dict(err_values)


    print(stats.get())
    print(stats.get_mean())


if __name__ == '__main__':
    main()