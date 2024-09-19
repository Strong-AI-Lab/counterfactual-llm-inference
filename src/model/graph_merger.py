
import abc
from typing import List, Set, Dict, Tuple, Any
import networkx as nx


class GraphMerger(abc.ABC):
        
        @abc.abstractmethod
        def _find_similar_nodes(self, graphs : List[nx.DiGraph]) -> List[Dict[Tuple[str,Any]]]:
            pass

        def merge_graphs(self, graphs : List[nx.DiGraph]) -> nx.DiGraph:
            # Find similar nodes
            similar_nodes = self._find_similar_nodes(graphs)

            # Update node names
            graph_renamed_copies = []
            for i, (graph, node_mapping) in enumerate(zip(graphs, similar_nodes)):
                name_mapping = {old_name: new_name for old_name, (new_name, _) in node_mapping.items()}
                attrs_mapping = {new_name: attrs for _, (new_name, attrs) in node_mapping.items()}

                new_graph = nx.relabeled_nodes(graph, name_mapping)
                for node, attrs in attrs_mapping.items():
                    new_graph.nodes[node].update(attrs)
                graph_renamed_copies.append(new_graph)

            # Merge graphs
            merged_graph = nx.DiGraph()
            for graph in graph_renamed_copies:
                merged_graph = nx.compose(merged_graph, graph)

            return merged_graph

        def __call__(self, graphs : List[nx.DiGraph]) -> nx.DiGraph:
            return self.merge_graphs(graphs)



MERGERS = {
}