
from typing import Any, Set, Type, Optional, List, Tuple, Dict
import numpy as np
import networkx as nx

from src.model.singleton import Singleton
from src.model.graph_builder import GraphBuilder
from src.model.graph_merger import GraphAbstractionMerger, GraphAnalogyMerger
from src.model.inference_oracle import InferenceOracle
from src.model.interpreter import Interpreter
from src.model.evaluator import Evaluator

import pydantic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from sklearn.cluster import DBSCAN



class LangchainSingleton(Singleton):

    def __init__(self, model_type : str, **kwargs):
        super().__init__(model_type=model_type, **kwargs)
        self.model_type = model_type

    def _create_instance(self, model_type, **kwargs):
        if model_type == 'openai':
            return ChatOpenAI(**kwargs)
        elif model_type == 'google':
            return ChatVertexAI(**kwargs)
        elif model_type == 'anthropic':
            return ChatAnthropic(**kwargs)
        elif model_type == 'mistral':
            return ChatMistralAI(**kwargs)
        elif model_type == 'ollama':
            return ChatOllama(**kwargs)
        else:
            raise ValueError(f"Model {model_type} not supported.")

    def _answer(self, system_message : str, user_message : str, structured_output_class : Optional[Type] = None):
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        model = self.instance

        if structured_output_class is not None:
            model = model.with_structured_output(structured_output_class)
        
        response = model.invoke(messages)

        return response


class LangchainEmbeddingsSingleton(Singleton):

    def _create_instance(self, model_type, **kwargs):
        if model_type == 'openai_embeddings':
            return OpenAIEmbeddings(**kwargs)
        elif model_type == 'google_embeddings':
            return VertexAIEmbeddings(**kwargs)
        elif model_type == 'mistral_embeddings':
            return MistralAIEmbeddings(**kwargs)
        elif model_type == 'ollama_embeddings':
            return OllamaEmbeddings(**kwargs)
        else:
            raise ValueError(f"Model {model_type} not supported.")
        
    def _embed(self, text : str) -> np.ndarray:
        embeddings = self.instance.embed_query(text) # [D]
        return np.array(embeddings)
    
    def _embed_batch(self, texts : List[str]) -> np.ndarray:
        embeddings = self.instance.embed_documents(texts) # [B,D]
        return np.array(embeddings)





class LangchainGraphBuilderSingleton(LangchainSingleton, GraphBuilder):
    class CausalVariable(pydantic.BaseModel):
        """ Attributes of a causal variable """
        
        description : str = pydantic.Field(description="A high-level short atomic description of the causal variable.", default='')
        type : str = pydantic.Field(description="The type of the variable (e.g. bool, int, set element, range element).", default='')
        values : str = pydantic.Field(description="The set of possible values, if applicable.", default='')
        current_value : str = pydantic.Field(description="The current instanciation of the variable.", default='')
        context : str = pydantic.Field(description="Additional contextual information linked to the current instance.", default='')

    class CausalEdge(pydantic.BaseModel):
        """ Attributes of a causal relationship """
        
        description : str = pydantic.Field(description="A high-level short atomic description of the causal relationship from the source variable to the target variable.", default='')
        details : str = pydantic.Field(description="A detailled explanation of how the value of the source variable and affects the value of the target variable in the text.", default='')

    class CausalGraphAtttributes(pydantic.BaseModel):
        """ Attributes of the causal graph """

        observed_nodes : List[Tuple[str, 'LangchainGraphBuilderSingleton.CausalVariable']] = pydantic.Field(description="List of observed causal variables. Each entry is a tuple with the node ID and a dictionary of attributes." \
                                                                        "The node ID is a unique string identifier for the causal variable, e.g. '0', '1', etc. " \
                                                                        "The dictionary of attributes must contain the following fields: " \
                                                                        "`description`: (str) a high-level short atomic description of causal variable; " \
                                                                        "`type`: (str) the type of the variable (e.g. bool, int, set element, range element); " \
                                                                        "`values`: (str) the set of possible values, if applicable; " \
                                                                        "`current_value`: (str) the current instanciation of the variable; " \
                                                                        "`context`: (str) additional contextual information linked to the current instance. "
                                                                         )
        hidden_nodes : List[Tuple[str, 'LangchainGraphBuilderSingleton.CausalVariable']] = pydantic.Field(description="List of hidden causal variables. Each entry is a tuple with the node ID and a dictionary of attributes." \
                                                                        "The node ID is a unique string identifier for the causal variable, e.g. 'h0', 'h1', etc. " \
                                                                        "The dictionary of attributes is the same as for the observed nodes except that the `current_value` field is left empty." \
                                                                        )
        observed_edges : List[Tuple[str,str,'LangchainGraphBuilderSingleton.CausalEdge']] = pydantic.Field(description="List of observed causal relationships. Each entry is a tuple with the source node ID, the target node ID and a dictionary of attributes." \
                                                                        "The source and target node IDs are the unique string identifiers of existing causal variables. Variables cannot be hidden variables. " \
                                                                        "The dictionary of attributes must contain the following fields: " \
                                                                        "`description`: (str) a high-level short atomic description of the causal relationship from the source variable to the target variable; " \
                                                                        "`details`: (str) A detailled explanation of how the value of the source variable and affects the value of the target variable in the text. " \
                                                                        )
        hidden_edges : List[Tuple[str,str,'LangchainGraphBuilderSingleton.CausalEdge']] = pydantic.Field(description="List of hidden causal relationships. Each entry is a tuple with the source node ID, the target node ID and a dictionary of attributes." \
                                                                        "The source and target node IDs are the unique string identifiers of existing causal variables. Source variables must be hidden. Target variables must be observed. " \
                                                                        "The dictionary of attributes is the same as for the observed edges." \
                                                                        )
    
    def _build_prompt(self, text : str) -> Tuple[str,str]:
        system_prompt = "Your task is to summarise a text into a list of instantiated causal variables and write down the causal relationships between them. " \
                "Variables should be as atomic and detailled as possible. Causal relationships should describe how the value of the first variable affects the value of the second. " \
                "One sentence usually describes two or more variables and connects them. For each variable, ask 'what are the causes of this variable's value? " \
                "Is it fully explained by the available information or are some causes missing?'. " \
                "If some causes seem to be missing, create new (hidden) variables. Hidden variables represent missing information to fully explain the value of one or more observed variables. " \
                "Hidden variables cannot have incoming edges. Reason step-by-step. Start by describing the content of the text snippet, explain it with your own words, identify the major and minor variables and how they are connected. " \
                "Answer the questions. Add the missing unknown variables when necessary. Follow carefully the instructions. Then, write down your answer using the given format very strictly."
        
        prompt = f"Here is the input text: \n\n```\n{text}\n```\n"

        return system_prompt, prompt

    def _parse_text(self, text_name: str, text: str) -> dict:
        system_prompt, prompt = self._build_prompt(text)
        answer = self._answer(system_prompt, prompt, LangchainGraphBuilderSingleton.CausalGraphAtttributes)
        
        return answer.dict()
    



class LangchainGraphAbstractionMergerSingleton(LangchainEmbeddingsSingleton, GraphAbstractionMerger):

    def __init__(self, model_type : str, eps : float = 0.5, message_propagation : int = 1, **kwargs):
        super().__init__(model_type, **kwargs)
        self.eps = eps
        self.message_propagation = message_propagation
        self.clustering_algorithm = DBSCAN(eps=self.eps)

    def _get_node_str(self, node : Dict[str,str]) -> str:
        return f"description: {node['description']}, type: {node['type']}, values: {node['values']}, context: {node['context']}"

    def _build_node_inputs(self, graph : nx.DiGraph) -> List[str]:
        node_inputs = []
        for node, attrs in graph.nodes(data=True):
            node_str = self._get_node_str(attrs)
            if self.message_propagation > 0:
                neighbours = nx.single_source_shortest_path_length(graph, node, cutoff=self.message_propagation)
                node_str += '\n' + '\n'.join([f'neighbour at distance {rank} from node: {self._get_node_str(graph.nodes[n])}' for n, rank in neighbours.items() if n != node])
            node_inputs.append(node_str)
        return node_inputs
    
    def _find_similar_nodes(self, graphs : List[nx.DiGraph]) -> List[Dict[str,Tuple[str,Any]]]:
        # Generate embeddings and similarity matrix
        node_inputs = [node_input for graph in graphs for node_input in self._build_node_inputs(graph)]
        embeddings = self._embed_batch(node_inputs) # [N,D] with N = N_g0 + N_g1 + ... + N_gn

        # Build clusters
        clustering = self.clustering_algorithm.fit(embeddings)
        labels = clustering.labels_
        cores = clustering.core_sample_indices_

        # Build mapping
        reversed_labels_dict = {l : [] for l in range(len(cores))}
        for i, label in enumerate(labels):
            if label != -1:
                reversed_labels_dict[label].append(i)

        core_attrs = []
        graph_idxs = [i for i, graph in enumerate(graphs) for _ in range(len(graph.nodes))]
        nodes_idxs = [i for graph in graphs for i in graph.nodes]
        for i in cores:
            graph_idx = graph_idxs[i]
            node_idx = nodes_idxs[i]

            attrs = graphs[graph_idx].nodes[node_idx]
            cluster_attrs = [graphs[graph_idxs[j]].nodes[nodes_idxs[j]] for j in reversed_labels_dict[i]]
            attrs['values'] = '; '.join([attrs['values']] + [attr['values'] for attr in cluster_attrs])
            attrs['context'] = '; '.join([attrs['context']] + [attr['context'] for attr in cluster_attrs])
            if not attrs['observed']:
                attrs['observed'] = any([attr['observed'] for attr in cluster_attrs])

            core_attrs.append((f'c{i}-{node_idx}', attrs))

        # Update similar nodes
        similar_nodes = []
        offset_idx = 0
        for graph in graphs:
            node_mapping = {}
            for j, node in enumerate(graph.nodes):
                label = labels[offset_idx+j]
                if label != -1:
                    node_mapping[node] = core_attrs[label]
            similar_nodes.append(node_mapping)
            offset_idx += len(graph.nodes)

        return similar_nodes
    

class LangchainGraphAnalogyMergerSingleton(LangchainEmbeddingsSingleton, GraphAnalogyMerger):

    def __init__(self, model_type : str, eps : float = 0.5, message_propagation : int = 1, **kwargs):
        super().__init__(model_type, **kwargs)
        self.eps = eps
        self.message_propagation = message_propagation
        self.clustering_algorithm = DBSCAN(eps=self.eps)

    def _get_node_str(self, node : Dict[str,str]) -> str:
        return f"description: {node['description']}, type: {node['type']}, values: {node['values']}, context: {node['context']}"

    def _build_node_inputs(self, graph : nx.DiGraph) -> List[str]:
        node_inputs = []
        for node, attrs in graph.nodes(data=True):
            node_str = self._get_node_str(attrs)
            if self.message_propagation > 0:
                neighbours = nx.single_source_shortest_path_length(graph, node, cutoff=self.message_propagation)
                node_str += '\n' + '\n'.join([f'neighbour at distance {rank} from node: {self._get_node_str(graph.nodes[n])}' for n, rank in neighbours.items() if n != node])
            node_inputs.append(node_str)
        return node_inputs

    def _find_analogical_nodes(self, graphs: List[nx.DiGraph]) -> List[Set[str]]:
        # Generate embeddings and similarity matrix
        node_inputs = [node_input for graph in graphs for node_input in self._build_node_inputs(graph)]
        embeddings = self._embed_batch(node_inputs)

        # Build clusters
        clustering = self.clustering_algorithm.fit(embeddings)
        labels = clustering.labels_

        # Build analogical sets
        node_idxs = [node for graph in graphs for node in range(len(graph.nodes))]
        analogical_nodes = [set() for _ in range(len(clustering.core_sample_indices_))]
        for i, label in enumerate(labels):
            if label != -1:
                analogical_nodes[label].add(node_idxs[i])

        return analogical_nodes






class LangchainInferenceOracleSingleton(LangchainSingleton, InferenceOracle):
    class CausalValue(pydantic.BaseModel):
        """ Estimated value, confidence and explanation of a causal variable """
        
        estimated_value : str = pydantic.Field(description="The current instanciation of the variable.", default='')
        confidence : float = pydantic.Field(description="The confidence of the model in the estimated value.", default=-1.0)
        explanation : str = pydantic.Field(description="The explanation of the estimated value given the attributes and values of the parent causes.", default='')

    def _build_node_description(self, node_attributes : Dict[str,str]) -> str:
        descr = f"description: {node_attributes['description']}, type: {node_attributes['type']}"
        if 'values' in node_attributes and node_attributes['values'] and len(node_attributes['values']) > 0: 
            descr += f", possible values: {node_attributes['values']}"
        descr += f", context: {node_attributes['context']}"

        return descr

    def _build_prompt(self, target_node_attributes : Dict[str,str], parent_node_attributes : List[Dict[str,str]], edge_attributes : List[Dict[str,str]]) -> Tuple[str,str]:
        system_prompt = "Your task is to predict the value of the target variable given its description, type, possible values and context, and the atributes and values of its parent causes and the relationships connecting them. " \
                        "The value of the target variable is fully determined by its direct list of causes. " \
                        "Reason step-by-step. Start by describing the attributes of the target variable and explain with your own words its relationships with its parent causes, how the variables are linked and how their values cause the value of the target." \
                        "Then, predict the value of the target variable. Provide a confidence score as a float between 0 and 1. Follow strictly the provided format."
                        
        prompt = f"The target variable has the following attributes: {self._build_node_description(target_node_attributes)}\nIt is caused by the following variables:\n"
        for i, (node, edge) in enumerate(zip(parent_node_attributes, edge_attributes)):
            prompt += f"{i}. {self._build_node_description(node)}. Its value is {node['updated_value']}. Its causal relationship with the target is described as follows: {edge['description']}\n"
        prompt += "Predict the value of the target variable."

        return system_prompt, prompt
    
    def predict(self, target_node_attributes : Dict[str,str], parent_node_attributes : List[Dict[str,str]], edge_attributes : List[Dict[str,str]]) -> Tuple[str, float, str]:
        system_prompt, prompt = self._build_prompt(target_node_attributes, parent_node_attributes, edge_attributes)
        answer = self._answer(system_prompt, prompt, LangchainInferenceOracleSingleton.CausalValue)
        
        return answer.estimated_value, answer.confidence, answer.explanation




class LangchainInterpreterSingleton(LangchainSingleton, Interpreter):
    class AlternativeValue(pydantic.BaseModel):
        """ Alternative/counterfactual value of a causal variable """
        
        factual_value : str = pydantic.Field(description="The current instanciation of the variable.", default='')
        counterfactual_value : str = pydantic.Field(description="The counterfactual instanciation of the variable.", default='')
        explanation : str = pydantic.Field(description="The explanation for the choice of the counterfactual value.", default='')

    def _build_prompt(self, attrs : Dict[str, Any]) -> Tuple[str,str]:
        system_prompt = "Your task is to interpret the attributes of a variable and propose an alternative/counterfactual instantiation different from its current value. " \
                        "The variable is described by its description, type, possible values, current value and context. " \
                        "The counterfactual value should be a plausible alternative instantiation of the variable given the context, type, description and posible values. " \
                        "Reason step-by-step. Start by describing the attributes of the variable and explain with your own words the reasons for the choice of the counterfactual value. " \
                        "Then, state the factual value and propose the new counterfactual value. Provide a confidence score as a float between 0 and 1. Follow strictly the provided format."
        
        prompt = f"The variable has the following attributes: description: {attrs['description']}, type: {attrs['type']}"
        if 'values' in attrs and attrs['values'] and len(attrs['values']) > 0: 
            prompt += f", possible values: {attrs['values']}"
        prompt += f", context: {attrs['context']}. The current value is {attrs['current_value']}. Propose a counterfactual value."

        return system_prompt, prompt
    
    def interpret(self, attrs: Dict[str, Any]) -> str:
        system_prompt, prompt = self._build_prompt(attrs)
        answer = self._answer(system_prompt, prompt, LangchainInterpreterSingleton.AlternativeValue)
        
        return answer.counterfactual_value
        



class LangchainEvaluatorSingleton(LangchainSingleton, Evaluator):
    class GraphEvaluation(pydantic.BaseModel):
        """ Evaluation of a causal graph """
        
        score : float = pydantic.Field(description="The score of the evaluation.", default=-1.0)
        confidence : float = pydantic.Field(description="The confidence of the model in the evaluation.", default=-1.0)
        explanation : str = pydantic.Field(description="The explanation of the evaluation.", default='')

    def _build_prompt(self, graph : nx.DiGraph) -> Tuple[str,str]:
        system_prompt = "Your task is to evaluate the plausibility of a set of events linked by causal relationships. " \
                        "The events are described by a high-level description and a value. The events are linked by causal relationships. The causal relationships are described by a high-level description. " \
                        "The overall plausibility of the set of events corresponds to the factorisation of the plausibility of the each event's occurence and given its causes. " \
                        "Reason step-by-step. Start by describing the events and the causal relationships. Explain with your own words the reasons for the plausibility of each event. " \
                        "Finally, provide an overall score for the plausibility of the sequence of events. Give an explanation describing your reasoning. Provide an overall confidence score as a float between 0 and 1. Follow strictly the provided format."
        
        ordered_nodes = list(nx.topological_sort(graph))
        order = {node: i for i, node in enumerate(ordered_nodes)}

        prompt = "The causal graph is composed of the following events:\n"
        for i, node in enumerate(ordered_nodes):
            for s, _, attrs in graph.in_edges(node, data=True):
                prompt += f"({order[s]} -> {i}) {attrs['description']}.\n"
            prompt += f"{i}. {graph.nodes[node]['description']}. The value is {graph.nodes[node]['current_value']}\n"

        return system_prompt, prompt

    def evaluate(self, graph : nx.DiGraph) -> Tuple[float, float, str]:
        system_prompt, prompt = self._build_prompt(graph)
        answer = self._answer(system_prompt, prompt, LangchainEvaluatorSingleton.GraphEvaluation)
        
        return answer.score, answer.confidence, answer.explanation



