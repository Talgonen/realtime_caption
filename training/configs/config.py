"""
YAML-based Configuration Loader for Hybrid Vision-Language Model
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigDict(dict):
    """Dictionary that allows attribute-style access."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def load_config(config_path: str) -> ConfigDict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ConfigDict with configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to ConfigDict for attribute access
    config = ConfigDict(config_dict)
    
    # Validate configuration
    _validate_config(config)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration saved to {save_path}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries recursively.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def _validate_config(config: ConfigDict):
    """Validate configuration parameters."""
    # Validate training parameters
    if 'training' in config:
        if config.training.get('batch_size', 1) < 1:
            raise ValueError("batch_size must be >= 1")
        if config.training.get('learning_rate', 1e-4) <= 0:
            raise ValueError("learning_rate must be > 0")
        if config.training.get('num_epochs', 1) < 1:
            raise ValueError("num_epochs must be >= 1")


def load_default_config() -> ConfigDict:
    """Load the default configuration."""
    config_dir = Path(__file__).parent
    default_config_path = config_dir / "default_config.yaml"
    return load_config(str(default_config_path))


def load_small_config() -> ConfigDict:
    """Load small-scale training configuration."""
    config_dir = Path(__file__).parent
    small_config_path = config_dir / "small_config.yaml"
    return load_config(str(small_config_path))


def load_large_config() -> ConfigDict:
    """Load large-scale training configuration."""
    config_dir = Path(__file__).parent
    large_config_path = config_dir / "large_config.yaml"
    return load_config(str(large_config_path))


# Backward compatibility aliases
get_default_config = load_default_config
get_small_config = load_small_config
get_large_config = load_large_config


if __name__ == "__main__":
    # Example: Print default configuration
    import json
    
    config = load_default_config()
    print("Default Configuration:")
    print(json.dumps(dict(config), indent=2))
    
    print("\n" + "="*60)
    print("Available configurations:")
    print("  - default_config.yaml")
    print("  - small_config.yaml")
    print("  - large_config.yaml")
    print("="*60)
