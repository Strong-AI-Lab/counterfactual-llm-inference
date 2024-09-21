
import abc
from typing import Dict, Any


def handle_config_singleton(config : Dict[str,Any]) -> Dict[str,Any]:
    """
    Handles the config for the singleton pattern. Duplicates parametrs shared by multiple singleton classes.

    Parameters
    ----------
    config : Dict[str,Any]
        The config to handle

    Returns
    -------
    Any
        The modified config
    """
    keys_considered = [key for key, value in config.items() if key.endswith("_class") and value.endswith("_singleton")]
    
    if len(keys_considered) > 0:
        config_key = keys_considered[0].replace("_class", "_config")

        if config_key in config and config[config_key] is not None:
            shared_config = config[config_key]

            for key in keys_considered[1:]:
                new_config_key = key.replace("_class", "_config")
                if new_config_key not in config or config[new_config_key] is None:
                    config[new_config_key] = shared_config

    return config


class Singleton(abc.ABC):
    _instances = {}

    @abc.abstractmethod
    def _create_instance(self):
        pass
    
    def __init__(self, model_type, *args, **kwargs):
        self.model_type = model_type
        if model_type not in Singleton._instances:
            Singleton._instances[model_type] = self._create_instance(model_type, *args, **kwargs)

    @property
    def instance(self):
        return Singleton._instances[self.model_type]