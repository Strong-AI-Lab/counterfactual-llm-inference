
from src.model.graph_builder import GPTGraphBuilder
from src.model.inference_oracle import DummyOracle




BUILDERS = {
    'gpt' : GPTGraphBuilder
}



MERGERS = {
}



INFERENCE_ORACLES = {
    "dummy": DummyOracle
}



INTERPRETERS = {

}



EVALUATORS = {
    
}