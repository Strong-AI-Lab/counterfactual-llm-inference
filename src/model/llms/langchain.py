
from typing import Any, Set, Type, Optional, List, Tuple, Dict
import numpy as np
import networkx as nx

from src.model.singleton import Singleton
from src.model.graph_builder import GraphBuilder
from src.model.graph_merger import GraphAbstractionMerger, GraphAnalogyMerger
from src.model.inference_oracle import InferenceOracle
from src.model.interpreter import NodeInterpreter, QueryInterpreter
from src.model.evaluator import Evaluator

import pydantic
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import StringPromptValue
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

    def __init__(self, model_type : str, max_retries : int = 7, **kwargs):
        super().__init__(model_type, **kwargs)
        self.max_retries = max_retries

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
            model = model.with_structured_output(structured_output_class, include_raw=True)
            answer = model.invoke(messages)

            if 'parsed' in answer and answer['parsed'] is not None and answer['parsing_error'] is None: # If the parsing was successful
                response = answer['parsed']
            else: # Attempt to recover from parsing error
                fix_parser = RetryOutputParser.from_llm(parser=PydanticOutputParser(pydantic_object=structured_output_class), llm=self.instance, max_retries=self.max_retries)

                if 'tool_calls' in answer['raw'].response_metadata['message']: # If parsing failed on subparts of the structured output
                    response = answer['raw'].response_metadata['message']['tool_calls'][0]['function']['arguments'] # Works for all ChatModels
                elif 'content' in answer['raw'].response_metadata['message']: # If parsing failed on the whole structured output
                    response = answer['raw'].response_metadata['message']['content']
                else: # In case
                    response = answer['raw'].content

                response = fix_parser.parse_with_prompt(str(response), StringPromptValue(text=f"{system_message}\n\n{user_message}"))

        else:
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
    class CausalVariable(pydantic.BaseModel): # /!\ The descriptions in nested classes are not forwarded to the model
        """ Attributes of a causal variable """

        node_id : str | int = pydantic.Field(description="A unique string identifier for the causal variable, e.g. '0', '1', etc.")
        description : str = pydantic.Field(description="A high-level short atomic description of the causal variable.", default='')
        type : str = pydantic.Field(description="The type of the variable (e.g. bool, int, set element, range element).", default='')
        values : str | List[str | int | float] | Set[str | int | float] = pydantic.Field(description="The set of possible values, if applicable.", default='')
        current_value : str | int | float = pydantic.Field(description="The current instanciation of the variable.", default='')
        context : str = pydantic.Field(description="Additional extensive contextual information linked to the current instance.", default='')

    class CausalEdge(pydantic.BaseModel): # /!\ The descriptions in nested classes are not forwarded to the model
            """ Attributes of a causal relationship """

            source_node_id : str | int = pydantic.Field(description="The unique string identifier of the source variable.")
            target_node_id : str | int = pydantic.Field(description="The unique string identifier of the target variable.")
            description : str = pydantic.Field(description="A high-level short atomic description of the causal relationship from the source variable to the target variable.", default='')
            details : str = pydantic.Field(description="A detailled explanation of how the value of the source variable and affects the value of the target variable in the text.", default='')

    class CausalGraphAttributes(pydantic.BaseModel):

        """ Attributes of the causal graph """
        observed_nodes : List['LangchainGraphBuilderSingleton.CausalVariable'] = pydantic.Field(description="List of observed causal variables. Each entry is a dictionary containing the following items:\n" \
                                                                        "node_id : (str) A unique string identifier for the causal variable, e.g. '0', '1', etc.;\n" \
                                                                        "description : (str) A high-level short atomic description of the causal variable;\n" \
                                                                        "type : (str) the type of the variable (e.g. bool, int, set element, range element);\n" \
                                                                        "values : (str) The set of possible values, if applicable;\n" \
                                                                        "current_value : (str) The current instanciation of the variable;\n" \
                                                                        "context : (str) Additional extensive contextual information linked to the current instance. "
                                                                         )
        hidden_nodes : List['LangchainGraphBuilderSingleton.CausalVariable'] = pydantic.Field(description="List of hidden causal variables. Each entry is a dictionary containing the same items as the observed variables:\n" \
                                                                        "node_id : (str) A unique string identifier for the hidden causal variable, e.g. 'h0', 'h1', etc.;\n" \
                                                                        "description : (str) A high-level short atomic description of the causal variable;\n" \
                                                                        "type : (str) the type of the variable (e.g. bool, int, set element, range element);\n" \
                                                                        "values : (str) The set of possible values, if applicable;\n" \
                                                                        "current_value : (str) This field is left empty because the current value of the variable is unknown since the variable is hidden;\n" \
                                                                        "context : (str) Additional extensive contextual information linked to the current instance. "
                                                                        )
        observed_edges : List['LangchainGraphBuilderSingleton.CausalEdge'] = pydantic.Field(description="List of observed causal relationships. Each entry is a dictionary containing the following items:\n" \
                                                                        "source_node_id : (str) A unique string identifier for the source causal variable (must exist and be in `observed_nodes`);\n" \
                                                                        "target_node_id : (str) A unique string identifier for the target causal variable (must exist and be in `observed_nodes`);\n" \
                                                                        "description : (str) A high-level short atomic description of the causal relationship from the source variable to the target variable;\n" \
                                                                        "details : (str) A detailled explanation of how the value of the source variable and affects the value of the target variable in the text. "
                                                                        )
        hidden_edges : List['LangchainGraphBuilderSingleton.CausalEdge'] = pydantic.Field(description="List of hidden causal relationships. Each entry is a dictionary containing the same items as the observed edges:\n" \
                                                                        "source_node_id : (str) A unique string identifier for the source causal variable (must exist and be in `hidden_nodes`);\n" \
                                                                        "target_node_id : (str) A unique string identifier for the target causal variable (must exist and be in `observed_nodes`);\n" \
                                                                        "description : (str) A high-level short atomic description of the causal relationship from the source variable to the target variable;\n" \
                                                                        "details : (str) A detailled explanation of how the value of the source variable and affects the value of the target variable in the text. "
                                                                        )
    
    def _build_prompt(self, text : str) -> Tuple[str,str]:
        system_prompt = "Your task is to summarise a text into a json dictionary of instantiated causal variables and the causal relationships between them. The obtained graph should be acyclic, i.e. no feedback loops are allowed. " \
                "Variables should be as atomic and detailled as possible. Causal relationships should describe how the value of the first variable affects the value of the second. " \
                "One sentence usually describes two or more variables and connects them. For each variable, the wollowing questions should be answered: 'what are the causes of this variable's value? " \
                "Is it fully explained by the available information or are some causes missing?' If some causes seem to be missing, create new (hidden) variables. " \
                "Hidden variables represent missing information to fully explain the value of one or more observed variables. They cannot have incoming edges. " \
                "Identify the major and minor variables and how they are connected. Add the missing unknown variables when necessary. Follow carefully the instructions and write down your answer using only the given json format very strictly." \
                "The format is as follows: \n\n" \
                "{" \
                "\"observed_nodes\": [\n" \
                "    {\n" \
                "        \"node_id\": (str) \"0\",\n" \
                "        \"description\": (str) \"<high-level short atomic description of causal variable 0>\",\n" \
                "        \"type\": (str) \"<variable type: e.g. bool, int, set element, range element>\",\n" \
                "        \"values\": (str) \"<set of possible values, if applicable>\",\n" \
                "        \"current_value\": (str) \"<current value>\",\n" \
                "        \"context\": (str) \"<contextual information type> : <value of the contextual information linked to the current instance>\"\n" \
                "    },\n" \
                "    ...\n" \
                "],\n" \
                "\"hidden_nodes\": [\n" \
                "    {\n" \
                "        \"node_id\": (str) \"h0\",\n" \
                "        \"description\": (str) \"<high-level short atomic description of the hidden causal variable>\",\n" \
                "        \"type\": (str) \"<variable type: e.g. bool, int, set element, range element>\",\n" \
                "        \"values\": (str) \"<set of possible values, if applicable>\",\n" \
                "        \"current_value\": (str) \"\", # This field is left empty because the current value of the variable is unknown since the variable is hidden\n" \
                "        \"context\": (str) \"<contextual information type> : <value of the contextual information linked to the current instance>\"\n" \
                "    },\n" \
                "    ...\n" \
                "],\n" \
                "\"observed_edges\": [\n" \
                "    {\n" \
                "        \"source_node_id\": (str) \"0\",\n" \
                "        \"target_node_id\": (str) \"1\",\n" \
                "        \"description\": (str) \"<high-level short atomic description of the causal relationship from variable 0 to 1>\",\n" \
                "        \"details\": (str) \"<detailled explanation of how the value of variable 0 affects the value of variable 1 in the text>\"\n" \
                "    },\n" \
                "    ...\n" \
                "],\n" \
                "\"hidden_edges\": [\n" \
                "    {\n" \
                "        \"source_node_id\": (str) \"h0\",\n" \
                "        \"target_node_id\": (str) \"1\",\n" \
                "        \"description\": (str) \"<high-level short atomic description of the causal relationship from hidden variable 0 to 1>\",\n" \
                "        \"details\": (str) \"<detailled explanation of how the value of hidden variable 0 affects the value of variable 1 in the text>\"\n" \
                "    },\n" \
                "    ...\n" \
                "]\n" \
                "}\n"
                
        
        prompt = f"Here is the input text: \n\n```\n{text}\n```\n"

        return system_prompt, prompt
    
    def _fix_errors(self, causal_graph : 'LangchainGraphBuilderSingleton.CausalGraphAttributes') -> None: # /!\ The fix is performed in-place
        for node_list in [causal_graph.observed_nodes, causal_graph.hidden_nodes]: 
            for node in node_list:
                if not isinstance(node.node_id, str):
                    node.node_id = str(node.node_id)
                if not isinstance(node.current_value, str):
                    node.current_value = str(node.current_value)
                if not isinstance(node.values, str):
                    node.values = str(node.values)

        for edge_list in [causal_graph.observed_edges, causal_graph.hidden_edges]:
            for edge in edge_list:
                if not isinstance(edge.source_node_id, str):
                    edge.source_node_id = str(edge.source_node_id)
                if not isinstance(edge.target_node_id, str):
                    edge.target_node_id = str(edge.target_node_id)

    def _parse_text(self, text_name: str, text: str) -> dict:
        system_prompt, prompt = self._build_prompt(text)
        answer = self._answer(system_prompt, prompt, LangchainGraphBuilderSingleton.CausalGraphAttributes)
        self._fix_errors(answer)
        
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




class LangchainRandomInterpreterSingleton(LangchainSingleton, NodeInterpreter):
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
        answer = self._answer(system_prompt, prompt, LangchainRandomInterpreterSingleton.AlternativeValue)
        
        return answer.counterfactual_value
    



class LangchainTextInterpreterSingleton(LangchainSingleton, QueryInterpreter):
    class ParsedQuery(pydantic.BaseModel):
        """ Query parsed into a structured dictionary """
        
        target_variable : str = pydantic.Field(description="The name of the target node.", default='')
        intervention_variable : str = pydantic.Field(description="The name of the intervention node.", default='')
        intervention_new_value : str = pydantic.Field(description="The value of the intervention node after intervention.", default='')
        inntervention_old_value : str = pydantic.Field(description="The old value of the intervention node before intervention.", default='')

    def _build_prompt(self, text : str, nodes : Optional[List[Tuple[str,Dict[str,Any]]]] = None) -> Tuple[str,str]:
        system_prompt = "Your task is to interpret a prompt text asking the result of a counterfactual query. " \
                        "The prompt text is composed of a target variable, an intervention variable, the value of the intervention variable after the intervention and the value of the intervention variable before the intervention. " \
                        "It has the shape \"Would Y if if X' instead of X?\" where Y is the target variable, X is the intervention variable and its value before the intervention, and X' is the value of the intervention variable after the intervention. " \
                        "Be careful, X and its value may be stated in various way, e.g. \"not X\" if X is a boolean variable or \"X does ...\" if X is a variable with a specific value. " \
                        "If you are given the list of variables and their types, select the right target and intervention IDs from the list. Provide the values of the intervention variable before and after the intervention. " \
                        "Follow strictly the provided format."
        
        prompt = f"The prompt text is: \n```\n{text}\n```\n"
        if nodes is not None:
            str_nodes = [f"(ID: {idx} type: {attrs['type']} description: {attrs['description']})" for idx, attrs in nodes]
            prompt += f"The list of variables is: {'; '.join(str_nodes)}. From the list, select the correct target variable ID, the correct intervention variable ID, the value of the intervention variable after the intervention and the value of the intervention variable before the intervention."
        else:
            prompt += "Select the target variable, the intervention variable, the value of the intervention variable after the intervention and the value of the intervention variable before the intervention."
        
        return system_prompt, prompt
    
    def _parse_errors(self, answer : Dict[str,Any], nodes : Optional[List[str]]) -> Dict[str,Any]:
        node_descriptions = {attrs['description'].lower() : id for id, attrs in nodes}
        
        if answer['target_variable'].lower() in node_descriptions:
            answer['target_variable'] = node_descriptions[answer['target_variable'].lower()]

        if answer['intervention_variable'].lower() in node_descriptions:
            answer['intervention_variable'] = node_descriptions[answer['intervention_variable'].lower()]
        
        return answer
    
    def interpret(self, text : str, nodes : Optional[List[str]] = None) -> Dict[str, Any]:
        system_prompt, prompt = self._build_prompt(text, nodes)
        answer = self._answer(system_prompt, prompt, LangchainTextInterpreterSingleton.ParsedQuery)

        answer = answer.dict()

        if nodes is not None:
            answer = self._parse_errors(answer, nodes)

        return answer
        



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
                        "Reason step-by-step and ive an explanation describing your reasoning. Start by describing the events and the causal relationships. Explain with your own words the reasons for the plausibility of each event. " \
                        "Finally, provide an overall score as a float between 0 and 1 for the plausibility of the sequence of events. Provide an overall confidence score as a float between 0 and 1. Follow strictly the provided json format. " \
                        "The format is as follows: \n\n" \
                        "{" \
                        "\"explanation\": \"<explanation of the evaluation>\",\n" \
                        "\"score\": <float between 0 and 1>,\n" \
                        "\"confidence\": <float between 0 and 1>\n" \
                        "}\n"
        
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



