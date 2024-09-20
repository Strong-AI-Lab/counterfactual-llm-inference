
import os
import abc
import re
import time
from typing import List, Tuple, Dict, Optional
import networkx as nx

import openai




class GraphBuilder(abc.ABC):
    
    def build_graph(self, text_name : str, text : str) -> nx.Graph:
        parse_dict = self._parse_text(text_name, text)
        graph = self._dict_to_graph(**parse_dict)
        return graph
    
    @abc.abstractmethod
    def _parse_text(self, text_name : str, text : str) -> dict:
        pass

    def _dict_to_graph(self, observed_nodes : List[Tuple[str, Dict[str,str]]], 
                             hidden_nodes : List[Tuple[str, Dict[str,str]]],
                             observed_edges : List[Tuple[str,str,Dict[str,str]]],
                             hidden_edges : List[Tuple[str,str,Dict[str,str]]]) -> nx.Graph:
        graph = nx.DiGraph()

        for node, attributes in observed_nodes:
            graph.add_node(node, **attributes, observed=True)

        for node, attributes in hidden_nodes:
            graph.add_node(node, **attributes, observed=False)

        for source, target, attributes in observed_edges:
            graph.add_edge(source, target, **attributes, observed=True)

        for source, target, attributes in hidden_edges:
            graph.add_edge(source, target, **attributes, observed=False)

        return graph
    
    def __call__(self, text_name : str, text : str) -> nx.Graph:
        return self.build_graph(text_name, text)


        
class GPTGraphBuilder(GraphBuilder):

    def __init__(self, model : str = 'gpt-4o', api_key : Optional[str] = None, is_payload_data : bool = False, save_payload_to_cache : bool = False, cache_folder : str = 'cache') -> None:
        """
        model : str
            The model to use for the summarisation task. Default is 'gpt-4o'.
        api_key : str
            The OpenAI API key. If None, it will be extracted from the environment variable 'OPENAI_API
        is_payload_data : bool
            If True, the data is already in the proper format and does not need to be summarised. used when running the model a second timea nd extracting response from cache. Default is False.
        """

        if api_key is None:
            api_key = os.environ['OPENAI_API_KEY']

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.is_payload_data = is_payload_data
        self.save_payload_to_cache = save_payload_to_cache
        self.cache_folder = cache_folder

        # Regex to extract nodes and edges
        self.reg_text = re.compile(r"```(.*?)```", re.DOTALL)
        self.reg_events = re.compile(r"^\s*(\d+): (.*)\n",flags=re.MULTILINE)
        self.reg_missing_events = re.compile(r"^\s*(h\d+): (.*)\n",flags=re.MULTILINE)
        self.reg_relations = re.compile(r"^\s*(h?\d+)((?: & h?\d+)*) -> (\d+): (.*)\n",flags=re.MULTILINE)
        self.reg_add_relations = re.compile(r" & (h?\d+)")
        self.reg_node_value = re.compile(r"^\s*(?P<description>.*) \[(?P<type>.*)\](?: \((?P<values>.*)\))?(?: (?P<current_value>.*))?; \{(?P<context>.+?)\}\s*\n?$")
        self.reg_edge_value = re.compile(r"^\s*(?P<description>.*); (?P<details>.*)\s*\n?$")

    def _build_prompt(self, text : str) -> str:
        prompt = "Your task is to summarise a text into a list of instantiated causal variables and write down the causal relationships between them. Variables should be as atomic and detailled as possible. Causal relationships should describe how the value of the first variable affects the value of the second. " \
                "One sentence usually describes two or more variables and connects them. For each variable, ask 'what are the causes of this variable's value? Is it fully explained by the available information or are some causes missing?'. " \
                "If some causes seem to be missing, create new (hidden) variables for them with the format 'hi' as indicated below. Hidden variables represent missing information to fully explain the value of one or more observed variables. " \
                "Hidden variables cannot have incoming edges. " \
                "Reason step-by-step. Start by describing the content of the text snippet, explain it with your own words, identify the major and minor variables and how they are connected. Answer the questions. Add the missing unknown variables when necessary. Follow carefully the instructions. Then, write down your answer using the following format very strictly: \n" \
                "```\n" \
                "0: <high-level short atomic description of causal variable 0> [<variable type: bool, int, set element, range element>] (<set of possible values, if applicable>) <current value>; {<contextual information type> : <value of the contextual information linked to the current instance>} \n" \
                "1: <...> \n" \
                "... \n" \
                "n: <...> \n\n" \
                "h0: <high-level short atomic description of the hidden causal variable> [<variable type: bool, int, set element, range element>] (<set of possible values, if applicable>) <NO VALUE AS VARIABLE IS UNOBSERVED>; {<contextual information type> : <value of the contextual information linked to the current instance>} \n" \
                "h1: <...> \n" \
                "... \n" \
                "hk: <...> \n\n" \
                "1 -> 2: <high-level short atomic description of the causal relationship from variable 1 to 2>; <detailled explanation of how the value of variable 1 affects the value of variable 2 in the text> \n" \
                "2 & 5 -> 6: <high-level short atomic description of the causal relationship from variables 2 and 5 to 6>; <detailled explanation of how the values of variable 2 and 5 affect the value of variable 6 in the text> \n" \
                "h0 -> 7: <...> \n" \
                "h0 -> 8: <...> \n" \
                "... \n" \
                "h2 & 6 -> 5: <...> \n" \
                "7 -> 4: <...> \n" \
                "7 & 8 -> 12: <...> \n" \
                "```\n\n" \
                f"Here is the input text: \n\n```\n{text}\n```\n\nPlease follow the above instructions and summarise the text into a list of causal variables and causal relationships with the proper format."

        return prompt
    
    def _summarise(self, text : str) -> str:
        prompt = self._build_prompt(text)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    
    def _parse_value(self, value : str, regex : re.Pattern) -> Dict[str, str]:
        match_obj = regex.match(value)
        d = match_obj.groupdict(default='')
        return d

    def _parse_text(self, text_name : str, text : str) -> dict:
        if not self.is_payload_data: # If the data is not already in the proper format), summarise
            print('Summarising text...')
            summary = self._summarise(text)

            if self.save_payload_to_cache:
                print('Saving payload to cache...')
                os.makedirs(self.cache_folder, exist_ok=True)
                cache_path = os.path.join(self.cache_folder, f'{self.model}-{text_name}-summary-{time.strftime("%Y%m%d-%H%M%S")}.txt')
                with open(cache_path, 'w+') as f:
                    f.write(summary)
        
        else:
            print('Data is already in the proper format. Skipping summarisation.')
            summary = text

        # Extract nodes and edges
        txt = '\n\n'.join(self.reg_text.findall(summary))
        observed_nodes = self.reg_events.findall(txt)
        observed_nodes = list(map(lambda x: (x[0], self._parse_value(x[1], self.reg_node_value)), observed_nodes))
        
        hidden_nodes = self.reg_missing_events.findall(txt)
        hidden_nodes = list(map(lambda x: (x[0], self._parse_value(x[1], self.reg_node_value)), hidden_nodes))

        edges = self.reg_relations.findall(txt)

        observed_edges = []
        hidden_edges = []
        for edge in edges:
            h, opt, t, r = edge
            r = self._parse_value(r, self.reg_edge_value)

            if h.startswith('h'):
                hidden_edges.append((h, t, r))
            else:
                observed_edges.append((h, t, r))

            if len(opt) > 0:
                for o in self.reg_add_relations.findall(opt):
                    if o.startswith('h'):
                        hidden_edges.append((o, t, r))
                    else:
                        observed_edges.append((o, t, r))

        return {'observed_nodes' : observed_nodes,
                'hidden_nodes' : hidden_nodes,
                'observed_edges' : observed_edges,
                'hidden_edges' : hidden_edges}


