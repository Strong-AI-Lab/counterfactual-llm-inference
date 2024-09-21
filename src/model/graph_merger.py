
import abc
from typing import List, Set, Dict, Tuple, Any
import networkx as nx


class GraphMerger(abc.ABC):

    @abc.abstractmethod
    def merge_graphs(self, graphs : List[nx.DiGraph]) -> nx.DiGraph:
        pass



class GraphAbstractionMerger(GraphMerger):
        
    @abc.abstractmethod
    def _find_similar_nodes(self, graphs : List[nx.DiGraph]) -> List[Dict[str,Tuple[str,Any]]]:
        """
        Finds common nodes shared by the graphs.

        Parameters
        ----------
        graphs : List[nx.DiGraph]
            The graphs to compare

        Returns
        -------
        List[Dict[str,Tuple[str,Any]]]
            The mapping of similar nodes. Each entry of the list maps to an input graph. The keys are the old node names and the values are tuples of the new node names and the node attributes
        """
        pass

    def merge_graphs(self, graphs : List[nx.DiGraph]) -> nx.DiGraph:
        # Rename nodes
        renamed_graphs = []
        for i, graph in enumerate(graphs):
            renamed_graphs.append(nx.relabel_nodes(graph, {node: f"g{i}-{node}" for node in graph.nodes}))

        # Find similar nodes
        similar_nodes = self._find_similar_nodes(renamed_graphs)

        # Update node names
        graph_renamed_copies = []
        for i, (graph, node_mapping) in enumerate(zip(renamed_graphs, similar_nodes)):
            name_mapping = {old_name: new_name for old_name, (new_name, _) in node_mapping.items()}
            attrs_mapping = {new_name: attrs for _, (new_name, attrs) in node_mapping.items()}

            new_graph = nx.relabel_nodes(graph, name_mapping)
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



class GraphAnalogyMerger(GraphMerger):
    
    @abc.abstractmethod
    def _find_analogical_nodes(self, graphs : List[nx.DiGraph]) -> List[Set[str]]:
        """
        Finds common hidden variables representing analogical links between the graphs.

        Parameters
        ----------
        graphs : List[nx.DiGraph]
            The graphs to compare

        Returns
        -------
        List[Set[str]]
            The list of analogical mapping between nodes. Each entry of the list corresponds to an analogical mapping. Each set contains the names of the nodes that are analogically linked.
        """
        pass

    def merge_graphs(self, graphs : List[nx.DiGraph]) -> nx.DiGraph:
        # Rename nodes
        renamed_graphs = []
        for i, graph in enumerate(graphs):
            renamed_graphs.append(nx.relabel_nodes(graph, {node: f"g{i}-{node}" for node in graph.nodes}))

        # Find analogical nodes
        analogical_nodes = self._find_analogical_nodes(renamed_graphs)

        # Merge graphs
        merged_graph = nx.DiGraph()
        for graph in renamed_graphs:
            merged_graph = nx.compose(merged_graph, graph)

        # Add analogical links
        for i, analogical_nodes_set in enumerate(analogical_nodes):
            merged_graph.add_node("ha{i}", observed=False, description="Unknown analogical mechanism shared by all descenddants", type="", values="", current_value="", context="")
            for node in analogical_nodes_set:
                merged_graph.add_edge(node, "ha{i}", observed=False, description="Analogical link", details="")

        return merged_graph
            