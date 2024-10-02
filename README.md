 # Self-Counterfactual Learning with Large Language Models

This repository contains code for generating causal graphs from natural language textual data and performing counterfactual reasoning with large language models.

## Install

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

You can install the packages ina virtual environment by running the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

For evaluation, we use data from the [Cladder dataset](https://arxiv.org/abs/2312.04350). The data is downloaded automatically from the [Huggingface hub](https://huggingface.co/datasets/causal-nlp/CLadder).

We also use real-world data from news article via [EventRegistry](https://eventregistry.org/). the dataset is not publicly available as of now but th raw data can be downloaded from [here](https://www.eia.gov/finance/markets/crudeoil/spot_prices.php).


## Usage

You can build a causal graph from text data by running the following command. Use the provided template to create your own configuration file.

```bash
python build_graph.py config/build_graph.yaml
```

You can perform counterfactual reasoning by running the following command. Load a generated causal graph and use the provided template to create your own configuration file.

```bash
python compute_counterfactuals.py config/counterfactual.yaml
```

:warning: **Warning:** Inference computations does not handle colliders yet. The generated computation graph shows the structure with colliders but the result of inference does not integrate information from colliders. Colliders are v-structures X -> Z <- Y with Z observed, creating a dependency between X and Y. As of now, Z is ignored in the inference computations. In practice, modifying observations can lead to nonsensical causal graphs. Favour using intervention nodes and counterfactual setting as this is the way this repository is intended to be used. 


You can perform end-to-end inference by running the following command. Use the provided template to create your own configuration file.

```bash
python end_to_end.py config/end_to_end.yaml
```

Evaluate the model on the Cladder dataset by running the following command:
```bash
python evaluate.py config/cladder_evaluate.yaml
```



## Causal Graph Attributes

The built causal graphs have the following attributes:

Nodes:
 - `description`: [str] A description of the causal variable
 - `type`: [str] The type of the causal variable
 - `values`: [str] The possible values of the causal variable
 - `current_value`: [str] The current value of the causal variable
 - `context`: [str] The context in which the causal variable is defined
 - `observed`: [bool] Whether the causal variable is observed or not
 - `layer`: [int] (Optional) The layer of the causal variable in the topological causal graph (used for visualisation)
 - `updated_value`: [str] (Optional) The updated value of the causal variable after inference

Edges:
 - `description`: [str] A description of the causal relationship
 - `details`: [str] Additional details about the causal relationship
 - `observed`: [bool] Whether the causal relationship is observed or not
