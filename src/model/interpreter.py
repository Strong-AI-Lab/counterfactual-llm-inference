
import abc
from typing import Dict, Any, Optional, List


class NodeInterpreter(abc.ABC):

    @abc.abstractmethod
    def interpret(self, attrs : Dict[str, Any]) -> str:
        """
        Interprets the attributes of a node and proposes an alternative/counterfactual instantiation.

        Parameters
        ----------
        attrs : Dict[str, Any]
            The attributes of the node to interpret. Required are: `description`, `type`, `values`, `context` and `current_value`

        Returns
        -------
        str
            The proposed alternative/counterfactual instantiation of the node
        """
        pass

    def __call__(self, attrs : Dict[str, Any]) -> str:
        return self.interpret(attrs)


class QueryInterpreter(abc.ABC):

    @abc.abstractmethod
    def interpret(self, text : str, nodes : Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Interprets a text to propose a structured  dictionary of attributes.

        Parameters
        ----------
        text : str
            The text to interpret
        nodes : Optional[List[str]], optional
            The list of nodes to consider, by default None

        Returns
        -------
        Dict[str, Any]
            The structured dictionary of attributes
        """
        pass

    def __call__(self, text : str, nodes : Optional[List[str]] = None) -> Dict[str, Any]:
        return self.interpret(text, nodes)