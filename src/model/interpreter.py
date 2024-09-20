
import abc
from typing import Dict, Any


class Interpreter(abc.ABC):

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
        