
import abc
from typing import Tuple
import networkx as nx


class Evaluator(abc.ABC):
    
    @abc.abstractmethod
    def evaluate(self, graph : nx.DiGraph) -> Tuple[float, float, str]:
        """
        Evaluates the graph and returns a score

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to evaluate
            
        Returns
        -------
        float
            The score of the graph
        float
            The confidence of the score (between 0 and 1)
        str
            The explanation of the score
        """
        pass

    def __call__(self, graph : nx.DiGraph) -> Tuple[float, float, str]:
        return self.evaluate(graph)