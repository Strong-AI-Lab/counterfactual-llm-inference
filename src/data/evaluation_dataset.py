
import abc
import re
from typing import Dict, Any, Tuple, List, Union, Optional
import networkx as nx

import datasets


class EvaluationDataset(abc.ABC):
    
    def __init__(self, path : Optional[Union[str,List[str]]] = None) -> None:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass
    
    @abc.abstractmethod
    def __getitem__(self, idx) -> str:
        pass
    
    @abc.abstractmethod
    def __iter__(self):
        pass


class CladderDataset(EvaluationDataset):

    def _parse_text_graph(self, text : str) -> nx.DiGraph:
        graph = nx.DiGraph()
        text_without_instructions = re.match(r'.*?: (.*)', text).group(1)
        lines = text_without_instructions.split('. ')

        graph = nx.DiGraph()
        edge_reg = re.compile(r'(.*) has a direct effect on (.*)')
        for line in lines[:-1]:
            edge_match = edge_reg.match(line)
            if edge_match:
                cause = edge_match.group(1)
                effects = edge_match.group(2).split(' and ')

                cause = cause.lower()
                effects = [effect.lower() for effect in effects]

                if not graph.has_node(cause):
                    graph.add_node(cause)
                for effect in effects:
                    if not graph.has_node(effect):
                        graph.add_node(effect)
                    graph.add_edge(cause, effect)

        return graph

    def _parse_item(self, idx : int, item : Dict[str,Any]) -> Dict[str,Any]:
        result_item = {}
        result_item['name'] = str(item['id'])
        result_item['label'] = item['label']

        split_text = item['prompt'].split('. ')
        query = split_text[-1]
        text = '. '.join(split_text[:-1])

        result_item['text'] = text
        result_item['query'] = query

        gt_graph = self._parse_text_graph(item['prompt'])
        result_item['gt_graph'] = gt_graph

        result_item['question_property'] = item['question_property']

        return result_item
    
    def __init__(self, path : Optional[Union[str,List[str]]] = None) -> None:
        data = datasets.load_dataset('causal-nlp/Cladder')['full_v1.5_default']
        self.data = data.filter(lambda e : e['query_type'] == 'det-counterfactual')
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx : int | List[int]) -> Dict[str,Any]:
        if isinstance(idx, list):
            return [self._parse_item(i, self.data[i]) for i in idx]
        elif isinstance(idx, slice):
            return [self._parse_item(i, self.data[i]) for i in range(*idx.indices(len(self)))]
        else:
            return self._parse_item(idx, self.data[idx])
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    



EVALUATION_DATASETS = {
    'cladder' : CladderDataset
}