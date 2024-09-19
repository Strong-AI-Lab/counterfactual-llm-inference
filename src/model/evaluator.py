
import abc
import networkx as nx


class Evaluator(abc.ABC):
    
    @abc.abstractmethod
    def evaluate(self, graph : nx.DiGraph) -> float:
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
        """
        pass

    def __call__(self, graph : nx.DiGraph) -> float:
        return self.evaluate(graph)