import os
import yaml
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class AppConfig:
    """Thread-safe Singleton configuration loader for Velox Vision."""
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AppConfig, cls).__new__(cls, *args, **kwargs)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, config_path: str = "config/default.yaml"):
        if getattr(self, "_loaded", False):
            return
        
        self.config_path = config_path
        self._config_data: Dict[str, Any] = {}
        self.load()
        self._loaded = True

    def load(self) -> None:
        """Reads and parses the configuration YAML file from disk."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file '{self.config_path}' not found. Falling back to empty configurations.")
            self._config_data = {}
            return
            
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration settings from '{self.config_path}'")
        except Exception as e:
            logger.error(f"Error parsing configuration file '{self.config_path}': {e}", exc_info=True)
            self._config_data = {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """Accesses a nested configuration property using dot notation (e.g. 'detection.device')."""
        keys = key_path.split(".")
        current_val: Any = self._config_data
        for key in keys:
            if isinstance(current_val, dict):
                current_val = current_val.get(key)
            else:
                return default
        return current_val if current_val is not None else default

# Global thread-safe settings accessor instance
settings = AppConfig()
