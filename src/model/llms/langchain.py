
from typing import Any, Type, Optional, List, Tuple, Dict
import networkx as nx

from src.model.singleton import Singleton
from src.model.graph_builder import GraphBuilder
from src.model.graph_merger import GraphMerger
from src.model.inference_oracle import InferenceOracle
from src.model.interpreter import Interpreter
from src.model.evaluator import Evaluator

import pydantic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama



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
    



class LangchainGraphMergerSingleton(LangchainSingleton, GraphMerger):
    pass




class LangchainInferenceOracleSingleton(LangchainSingleton, InferenceOracle):
    class CausalValue(pydantic.BaseModel):
        """ Estimated value, confidence and explanation of a causal variable """
        
        estimated_value : str = pydantic.Field(description="The current instanciation of the variable.", default='')
        confidence : float = pydantic.Field(description="The confidence of the model in the estimated value.", default='')
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
    
    def predict(self, target_node_attributes : Dict[str,str], parent_node_attributes : List[Dict[str,str]], edge_attributes : List[Dict[str,str]]) -> str:
        system_prompt, prompt = self._build_prompt(target_node_attributes, parent_node_attributes, edge_attributes)
        answer = self._answer(system_prompt, prompt, LangchainInferenceOracleSingleton.CausalValue)
        
        return answer.estimated_value




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
        
        score : float = pydantic.Field(description="The score of the evaluation.", default=0.0)
        confidence : float = pydantic.Field(description="The confidence of the model in the evaluation.", default=0.0)
        explanation : str = pydantic.Field(description="The explanation of the evaluation.", default='')

        # event_scores : List[float] = pydantic.Field(description="The scores of the individual events.")
        # event_confidences : List[float] = pydantic.Field(description="The confidences of the model in the individual events scores.")
        # event_explanations : List[str] = pydantic.Field(description="The explanations of the individual eventss scores.")

    def _build_prompt(self, graph : nx.DiGraph) -> Tuple[str,str]:
        # system_prompt = "Your task is to evaluate the plausibility of a set of events linked by causal relationships. " \
        #                 "The events are described by a high-level description and a value. The events are linked by causal relationships. The causal relationships are described by a high-level description. " \
        #                 "The overall plausibility of the set of events corresponds to the factorisation of the plausibility of the each event's occurence and given its causes. " \
        #                 "Reason step-by-step. Start by describing the events and the causal relationships. Then, provide scores for each event. " \
        #                 "Explain with your own words the reasons for the plausibility of each event and provide confidence scores as a float between 0 and 1 for each evaluation score. " \
        #                 "Finally, provide an overall score for the plausibility of the set of events along with an explantion describing your reasoning. Provide an overall confidence score as a float between 0 and 1. Follow strictly the provided format."
        system_prompt = "Your task is to evaluate the plausibility of a set of events linked by causal relationships. " \
                        "The events are described by a high-level description and a value. The events are linked by causal relationships. The causal relationships are described by a high-level description. " \
                        "The overall plausibility of the set of events corresponds to the factorisation of the plausibility of the each event's occurence and given its causes. " \
                        "Reason step-by-step. Start by describing the events and the causal relationships. Explain with your own words the reasons for the plausibility of each event. " \
                        "Finally, provide an overall score for the plausibility of the set of events along with an explantion describing your reasoning. Provide an overall confidence score as a float between 0 and 1. Follow strictly the provided format."
        
        ordered_nodes = list(nx.topological_sort(graph))
        order = {node: i for i, node in enumerate(ordered_nodes)}

        prompt = "The causal graph is composed of the following events:\n"
        for i, node in enumerate(ordered_nodes):
            for s, _, attrs in graph.in_edges(node, data=True):
                prompt += f"({order[s]} -> {i}) {attrs['description']}.\n"
            prompt += f"{i}. {graph.nodes[node]['description']}. The value is {graph.nodes[node]['current_value']}\n"

        return system_prompt, prompt

    def evaluate(self, graph : nx.DiGraph) -> float:
        system_prompt, prompt = self._build_prompt(graph)
        answer = self._answer(system_prompt, prompt, LangchainEvaluatorSingleton.GraphEvaluation)
        
        return answer.score



