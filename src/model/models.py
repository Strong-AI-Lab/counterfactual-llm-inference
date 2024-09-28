
from src.model.graph_builder import GPTGraphBuilder
from src.model.inference_oracle import DummyOracle
from src.model.llms.langchain import (
    LangchainGraphBuilderSingleton, 
    LangchainGraphAbstractionMergerSingleton, 
    LangchainGraphAnalogyMergerSingleton, 
    LangchainInferenceOracleSingleton, 
    LangchainRandomInterpreterSingleton, 
    LangchainTextInterpreterSingleton,
    LangchainEvaluatorSingleton
)



BUILDERS = {
    'gpt' : GPTGraphBuilder,
    'langchain_singleton' : LangchainGraphBuilderSingleton
}



MERGERS = {
    'langchain_abstraction_singleton' : LangchainGraphAbstractionMergerSingleton,
    'langchain_analogy_singleton' : LangchainGraphAnalogyMergerSingleton
}



INFERENCE_ORACLES = {
    "dummy": DummyOracle,
    'langchain_singleton' : LangchainInferenceOracleSingleton
}



NODE_INTERPRETERS = {
    'langchain_singleton' : LangchainRandomInterpreterSingleton
}



QUERY_INTERPRETERS = {
    'langchain_singleton' : LangchainTextInterpreterSingleton
}



EVALUATORS = {
    'langchain_singleton' : LangchainEvaluatorSingleton
}