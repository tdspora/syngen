from typing import Dict
from contextlib import contextmanager
import copy

class Context:
    def __init__(self):
        self.config = {}

    def set_config(self, value: Dict):
        self.config = copy.deepcopy(value)

    def get_config(self) -> Dict:
        return self.config

# Singleton pattern to ensure there is only one instance of Configuration
_config_instance: Context = None

def get_context() -> Context:
    global _config_instance
    if _config_instance is None:
        _config_instance = Context()
    return _config_instance


@contextmanager
def global_context(metadata: Dict) -> Context:
    global_config = get_context()
    global_config.set_config(metadata)

    yield global_config
