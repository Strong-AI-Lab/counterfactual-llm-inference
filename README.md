 # Self-Counterfactual Learning with Large Language Models

...

## Install


## Usage




:warning: **Warning:** Inference computations does not handle colliders yet. The generated computation graph shows the structure with colliders but the result of inference does not integrate information from colliders. Colliders are v-structures X -> Z <- Y with Z observed, creating a dependency between X and Y. As of now, Z is ignored in the inference computations. In practice, modifying observations can lead to nonsensical causal graphs. Favour using intervention nodes and counterfactual setting as this is the way this repository is intended to be used. 


## Causal Graph Attributes

The built causal graphs have the following attributes:

Nodes
 - `description`: [str] A description of the causal variable
 - `type`: [str] The type of the causal variable
 - `values`: [str] The possible values of the causal variable
 - `current_value`: [str] The current value of the causal variable
 - `context`: [str] The context in which the causal variable is defined
 - `observed`: [bool] Whether the causal variable is observed or not
 - `layer`: [int] (Optional) The layer of the causal variable in the topological causal graph (used for visualisation)
 - `updated_value`: [str] (Optional) The updated value of the causal variable after inference

Edges
 - `description`: [str] A description of the causal relationship
 - `details`: [str] Additional details about the causal relationship
 - `observed`: [bool] Whether the causal relationship is observed or not
