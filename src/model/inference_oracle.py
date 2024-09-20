
import abc
from typing import Dict, List


class InferenceOracle(abc.ABC):

    @abc.abstractmethod
    def predict(self, target_node_attributes : Dict[str,str], parent_node_attributes : List[Dict[str,str]], edge_attributes : List[Dict[str,str]]) -> str:
        """
        Predicts the value of the target node given the attributes of the target node, the parent nodes and the edges

        Parameters
        ----------
        target_node_attributes : Dict[str,str]
            The attributes of the target node. Required are: `description`, `type`, `values` and `context`
        parent_node_attributes : List[Dict[str,str]]
            The attributes of the parent nodes. Required are: `description`, `type`, `values`, `context` and `updated_value`
        edge_attributes : List[Dict[str,str]]
            The attributes of the edges. required are: `description` and `details`

        Returns
        -------
        str
            The predicted value of the target node
        """
        pass

    def __call__(self, target_node_attributes : Dict[str,str], parent_node_attributes : List[Dict[str,str]], edge_attributes : List[Dict[str,str]]) -> str:
        return self.predict(target_node_attributes, parent_node_attributes, edge_attributes)




class DummyOracle(InferenceOracle):

    def __init__(self):
        pass

    def predict(self, target_node_attributes : Dict[str,str], parent_node_attributes : List[Dict[str,str]], edge_attributes : List[Dict[str,str]]) -> str:
        return "dummy_value"

