
import abc
import networkx as nx
from typing import List


class GraphMerger(abc.ABC):
        
        @abc.abstractmethod
        def merge_graphs(self, graphs : List[nx.Graph]) -> nx.Graph:
            pass





MERGERS = {
}