
from typing import List, Union, Optional, Callable, Tuple
import networkx as nx


class Query():
    """
    Class for querying causal graphs. Observations can affect the value of the target nodes through direct or back-door paths. 
    Interventions only affect the value of the target nodes through direct paths. Counterfactual queries require to perform
    abduction and inference to estimate the required observations.
    """

    def __init__(self, graph : nx.Graph,
                 oracle : Callable[[str], str],
                 target_node : str, 
                 observation_nodes : Optional[Union[str, List[str]]] = None, 
                 intervention_nodes : Optional[Union[str, List[str]]] = None, 
                 observation_values : Optional[Union[str, List[str]]] = None, 
                 intervention_values : Optional[Union[str, List[str]]] = None, 
                 compute_counterfactuals : bool = False,
                 traversal_cutoff : Optional[int] = None
                 ) -> None:
        """
        graph : nx.Graph
            The causal graph to query.
        target_node : str
            The target node to estimate.
        observation_nodes : Optional[Union[str, List[str]]]
            The observation nodes to query, to be used for computing observatioal/interventional queries. If left unspecified, the query is assumed 
            counterfactual and the required observations are computed/estimated via abduction and inference. Default is None.
        observation_values : Optional[Union[str, List[str]]]
            The observation values to query, to be used for computing observational/interventional queries. The number of observation values must 
            correspond to the number of observation nodes. Default is None. The list can contain None values, the value of which will be extracted
            from the graph.
        intervention_nodes : Optional[Union[str, List[str]]]
            The intervention nodes to query, to be used for computing interventional or counterfactual queries. Default is None.
        intervention_values : Optional[Union[str, List[str]]]
            The intervention values to query, to be used for computing interventional or counterfactual queries. The number of intervention values 
            should correspond to the number of intervention nodes. Default is None. The list can contain None values, the value of which will be
            extracted from the graph.
        compute_counterfactuals : bool
            Whether to compute counterfactual queries. If True, the query is assumed to be counterfactual and the required observations are computed/estimated 
            via abduction and inference. If False, the query is assumed to be observational/interventional and the observations are assumed to be provided. 
            Default is False.
        traversal_cutoff : Optional[int]
            The maximum length of the paths to consider when computing causal paths. If None, no limit is applied. Default is None.
        """
        self.graph = graph
        self.oracle = oracle
        self.target_node = target_node
        self.observation_nodes = observation_nodes
        self.observation_values = observation_values
        self.intervention_nodes = intervention_nodes
        self.intervention_values = intervention_values
        self.compute_counterfactuals = compute_counterfactuals
        self.abduction_nodes = None
        self.traversal_cutoff = traversal_cutoff

        if observation_nodes is not None and observation_values is not None and len(observation_nodes) != len(observation_values):
            raise ValueError("The number of observation nodes and observation values must be equal.")
        
        if intervention_nodes is not None and intervention_values is None:
            raise ValueError("Intervention values must be provided if intervention nodes are specified.")
    
        if intervention_nodes is not None and intervention_values is not None and len(intervention_nodes) != len(intervention_values):
            raise ValueError("The number of intervention nodes and intervention values must be equal.")
        
        if self.compute_counterfactuals and self.intervention_nodes is None :
            raise ValueError("Intervention nodes must be provided for counterfactual queries.")
        
    
    def __call__(self) -> str:
        return self.estimate()
    
    def _node_repr(self, node : str, graph : nx.DiGraph) -> str:
        node_attrs = graph.nodes[node]
        return f"{node}: {', '.join([f'{key}={value}' for key, value in node_attrs.items()])}"

    def __repr__(self) -> str:
        query_type = ''
        if self.compute_counterfactuals:
            query_type = 'Counterfactual'
        elif self.intervention_nodes is not None:
            query_type = 'Interventional'
        else:
            query_type = 'Observational'

        query_repr = f"{query_type} query on {self.graph}{'' if self.traversal_cutoff is None else f' (limited graph traversal cutoff: {self.traversal_cutoff})'}."
        query_repr += f"\nTarget node:\n{self._node_repr(self.target_node, self.graph)}"
        if self.observation_nodes is not None:
            query_repr += f"\nObservation nodes:\n"
            query_repr += '\n'.join([self._node_repr(node, self.graph) for node in self.observation_nodes])
        if self.intervention_nodes is not None:
            query_repr += f"\nIntervention nodes:\n"
            query_repr += '\n'.join([self._node_repr(node, self.graph) for node in self.intervention_nodes])
        return query_repr
        
    def __str__(self) -> str:
        return f"Query(graph={self.graph}, " \
               f"target_node={self.target_node}, " \
               f"observation_nodes={self.observation_nodes}, " \
               f"observation_values={self.observation_values}, " \
               f"intervention_nodes={self.intervention_nodes}, " \
               f"intervention_values={self.intervention_values}, " \
               f"compute_counterfactuals={self.compute_counterfactuals}, " \
               f"traversal_cutoff={self.traversal_cutoff})"
    

    def estimate(self) -> str:
        """
        Estimate the target node value.

        Returns
        -------
        str
            The estimated target node value.
        """

        # Update graph with custom values
        graph = self._update_graph_values(self.graph)

        # If is counterfactual, perform abduction step
        if self.compute_counterfactuals:
            graph, abduction_nodes = self._compute_abduction(graph)
            self.abduction_nodes = abduction_nodes

        # Intervene on the graph
        graph = self._intervene(graph)
        
        # Build computation graph
        computation_graph = self._build_computation_graph(graph)

        # Estimate target node value
        conditioning_nodes = []
        if self.observation_nodes is not None:
            conditioning_nodes += self.observation_nodes
        if self.intervention_nodes is not None:
            conditioning_nodes += self.intervention_nodes
        if self.abduction_nodes is not None:
            conditioning_nodes += self.abduction_nodes
        value, inference_graph = self._estimate_target_value_from_parents(computation_graph, self.target_node, conditioning_nodes)

        return value, inference_graph


    def _update_graph_values(self, graph : nx.DiGraph) -> nx.DiGraph:
        """
        Update the graph with the observation values.

        Parameters
        ----------
        graph : nx.DiGraph
            The causal graph. Not modified.

        Returns
        -------
        nx.DiGraph
            The updated causal graph.
        """
        graph = graph.copy()

        if self.observation_nodes is not None and self.observation_values is not None:
            for node, value in zip(self.observation_nodes, self.observation_values):
                if value is not None:
                    graph.nodes[node]['current_value'] = value

        return graph
    

    def _compute_abduction(self, graph : nx.DiGraph) -> nx.DiGraph: # TODO: disambiguate how observed variables should act in counterfactal inference (see colliders in particular)
        """
        Compute the abduction step for counterfactual inference. Define the set of observation nodes that are required to compute the counterfactual query.
        If this set includes hidden variables U, estimate their values from the known variables X' in the factual world: P(U|X').

        Parameters
        ----------
        graph : nx.DiGraph
            The causal graph. Not modified.

        Returns
        -------
        nx.DiGraph
            The updated causal graph.
        """
        graph = graph.copy()

        if self.observation_nodes is None:
            observation_nodes = []

        # Get set of variables affected by the intervention
        intervention_paths = []
        intervention_graph = self._intervene(graph)
        for intervention_node in self.intervention_nodes:
            intervention_paths.extend(self._find_causal_paths(intervention_graph, intervention_node, self.target_node, observation_nodes))

        affected_variables = set([node for path in intervention_paths for node in path]) - set(observation_nodes) - set(self.intervention_nodes)

        # Get set of parents to update
        parents_to_update = set([parent for node in affected_variables for parent in graph.predecessors(node)])

        # Remove undesired nodes: observation nodes, intervention nodes, nodes on the path as they are not to be updated
        parents_to_update = parents_to_update - affected_variables - set(observation_nodes) - set(self.intervention_nodes)

        # Parents of nodes in the path must be considered as observed for counterfactual inference
        abduction_nodes = None
        if len(parents_to_update) > 0:
            abduction_nodes = list(parents_to_update)

            # If abduction node is hidden, estimate its value
            for node in abduction_nodes:
                if not graph.nodes[node]['observed']:
                    children = list(graph.successors(node))
                    subgraph = graph.subgraph([node] + children).reverse() # Subgraph of the node and its children, reverse to estimate value from children
                    inferred_value, _ = self._estimate_target_value_from_parents(subgraph, node, children)
                    graph.nodes[node]['current_value'] = inferred_value
        
        return graph, abduction_nodes
    
    
    def _intervene(self, graph : nx.DiGraph) -> nx.DiGraph:
        """
        Intervene on the graph by setting the removing the incoming edges of the intervention nodes and settin intervention values.

        Parameters
        ----------
        graph : nx.DiGraph
            The causal graph. Not modified.

        Returns
        -------
        nx.DiGraph
            The intervened causal graph.
        """
        graph = graph.copy()
        
        if self.intervention_nodes is not None:
            intervention_values =  self.intervention_values if self.intervention_values is not None else [None] * len(self.intervention_nodes)
            for node, value in zip(self.intervention_nodes, intervention_values):
                graph.remove_edges_from(list(graph.in_edges(node)))
                if value is not None:
                    graph.nodes[node]['current_value'] = value

        return graph
        

    def _build_computation_graph(self, graph : nx.DiGraph) -> nx.DiGraph:
        """
        Build the computation graph for the query. The computation graph is a subgraph of `self.graph` containing only the variables that require to be re-computed during inference.

        Parameters
        ----------
        graph : nx.DiGraph
            The causal graph. Not modified.

        Returns
        -------
        nx.DiGraph
            The computation graph.
        """
        
        known_nodes = set()
        if self.observation_nodes is not None:
            known_nodes.update(self.observation_nodes)
        if self.intervention_nodes is not None:
            known_nodes.update(self.intervention_nodes)
        if self.abduction_nodes is not None:
            known_nodes.update(self.abduction_nodes)

        nodes_to_keep = set(self.target_node)
        nodes_to_keep.update(known_nodes)

        for node in known_nodes:
            paths = self._find_causal_paths(graph, node, self.target_node, known_nodes - set([node]))
            nodes_to_keep.update(set([node for path in paths for node in path]))

        computation_graph = graph.subgraph(nodes_to_keep)
    
        computation_graph.nodes[self.target_node]['target'] = True
        if self.observation_nodes is not None:
            for observation_node in self.observation_nodes:
                computation_graph.nodes[observation_node]['observation'] = True
        if self.intervention_nodes is not None:
            for intervention_node in self.intervention_nodes:
                computation_graph.nodes[intervention_node]['intervention'] = True

        return computation_graph


    def _find_causal_paths(self, graph : nx.DiGraph, source : str, target : str, conditioning_nodes : Optional[List[str]] = None) -> List[List[str]]: # TODO: to optimise, filter out paths on the go
        """
        Find all causal paths between the source and target nodes.

        Parameters
        ----------
        graph : nx.DiGraph
            The causal graph. Not modified.
        source : str
            The source node.
        target : str
            The target node.
        conditioning_nodes : Optional[List[str]]
            The conditioning nodes.

        Returns
        -------
        List[List[str]]
            The list of causal paths.
        """
        paths = []

        if conditioning_nodes is None:
            conditioning_nodes = []

        candidate_paths = nx.all_simple_paths(graph.to_undirected(as_view=True), source, target, cutoff=self.traversal_cutoff) # Find all simple undirected paths between source and target nodes

        for path in candidate_paths:
            subgraph = graph.subgraph(path)
            sub_conditioning_nodes = set(path).intersection(set(conditioning_nodes))
            if not nx.is_d_separator(subgraph, source, target, sub_conditioning_nodes): # Filter out paths that are d-separated by the conditioning nodes
                paths.append(path)

        return paths


    def _estimate_target_value_from_parents_dyn(self, graph : nx.DiGraph, target_node : str, conditioning_nodes : List[str]) -> None:
        """
        Auxiliary function for target value estimation. Dynamically compute intermediate values. Updates the computation graph in place.

        Parameters
        ----------
        graph : nx.DiGraph
            The computation graph.
        target_node : str
            The target node.
        conditioning_nodes : List[str]
            The conditioning nodes.
        """
        if 'updated_value' in graph.nodes[target_node]: # Do not recompute target node value
            return

        if target_node in conditioning_nodes: # Do not update conditioning nodes
            graph.nodes[target_node]['updated_value'] = graph.nodes[target_node]['current_value']
            return

        if 'updated_value' not in graph.nodes[target_node]: # Otherwise, compute the target node value
            parents = list(graph.predecessors(target_node))
            for parent in parents:
                self._estimate_target_value_from_parents_dyn(graph, parent, conditioning_nodes) # Recursively compute parent values
            
            target_node_attrs = graph.nodes[target_node]
            parent_attrs = [graph.nodes[parent] for parent in parents]
            edge_attrs = [graph.edges[parent, target_node] for parent in parents]
            value, confidence, explanation = self.oracle(target_node_attrs, parent_attrs, edge_attrs) # Compute target node value with oracle

            graph.nodes[target_node]['updated_value'] = value
            graph.nodes[target_node]['updated_value_confidence'] = confidence
            graph.nodes[target_node]['updated_value_explanation'] = explanation

    def _estimate_target_value_from_parents(self, graph : nx.DiGraph, target_node : str, conditioning_nodes : List[str]) -> Tuple[str, nx.DiGraph]: # TODO: current version only computes from parents, suitable for counterfactuals but due to the abduction step but leaves out collider information in other cases. to integrate
        """
        Estimate the target node value given the conditioning nodes. Dynamically compute intermediate values.

        Parameters
        ----------
        graph : nx.DiGraph
            The computation graph. Not modified.
        target_node : str
            The target node.
        conditioning_nodes : List[str]
            The conditioning nodes.

        Returns
        -------
        str
            The estimated target node value.
        nx.DiGraph
            The updated computation graph.
        """
        graph = graph.copy()

        self._estimate_target_value_from_parents_dyn(graph, target_node, conditioning_nodes)
        value = graph.nodes[target_node]['updated_value']

        return value, graph
        