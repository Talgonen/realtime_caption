"""
Utility functions for the training pipeline
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple


def load_image(image_path: str, clip_preprocess=None) -> torch.Tensor:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        clip_preprocess: CLIP preprocessing function
        
    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    
    if clip_preprocess is not None:
        image = clip_preprocess(image)
    else:
        # Default preprocessing
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        image = transform(image)
    
    return image


def save_config(config, save_path: str):
    """Save configuration to JSON file."""
    import json
    from dataclasses import asdict
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"Configuration saved to {save_path}")


def load_config(config_path: str):
    """Load configuration from JSON file."""
    import json
    from training.configs.config import Config
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct config object
    # TODO: Implement proper deserialization
    return config_dict


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
