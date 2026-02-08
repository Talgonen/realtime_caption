#!/usr/bin/env python3
"""
Example usage of the RTVTLM model with adapters.
This script demonstrates how to use the model for image captioning and train the adapters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from training.models.rt_vlm import RTVTLM
from transformers import CLIPProcessor
from PIL import Image
import numpy as np

class DummyVisionLanguageDataset(Dataset):
    """Dummy dataset for demonstration purposes."""
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Generate dummy captions
        self.captions = [
            "A cat sitting on a chair",
            "A dog running in the park", 
            "A bird flying in the sky",
            "A car driving on the road",
            "A person walking down the street"
        ] * (num_samples // 5 + 1)
        self.captions = self.captions[:num_samples]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create dummy image
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Process image
        image_inputs = self.processor(images=dummy_image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].squeeze(0)
        
        caption = self.captions[idx]
        
        return {
            'pixel_values': pixel_values,
            'caption': caption
        }

def collate_fn(batch):
    """Custom collate function for the dataloader."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'captions': captions
    }

def example_usage():
    """Demonstrate model usage."""
    print("=== RTVTLM Model Example Usage ===\\n")
    
    # Initialize model with custom adapter positions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = RTVTLM(
        clip_model_name="openai/clip-vit-base-patch32",
        qwen_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        insert_positions=[2, 5, 8, 11, 14, 17, 20],  # More adapter positions
        device=device
    )
    
    # Print model info
    total_params, trainable_params = model.count_parameters()
    
    print(f"\\n=== Model Architecture ===")
    print(f"Adapter positions: {[2, 5, 8, 11, 14, 17, 20]}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Example inference
    print(f"\\n=== Inference Example ===")
    model.eval()
    
    # Create dummy input
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    prompt = "Describe this image:"
    
    # Tokenize input
    tokenized = model.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            images=dummy_image,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    print(f"Input prompt: '{prompt}'")
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    # Example training setup
    print(f"\\n=== Training Setup Example ===")
    
    # Create dummy dataset and dataloader
    dataset = DummyVisionLanguageDataset(num_samples=50)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Setup optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
    
    model.train()
    print(f"Created dataset with {len(dataset)} samples")
    print(f"DataLoader batch size: 2")
    print(f"Optimizer: AdamW with lr=1e-4")
    print(f"Only training {len(list(model.get_trainable_parameters()))} parameter tensors")
    
    # Example training step
    print(f"\\n=== Example Training Step ===")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx > 0:  # Just one example batch
            break
            
        pixel_values = batch['pixel_values'].to(device)
        captions = batch['captions']
        
        # Tokenize captions as labels
        tokenized_captions = model.tokenizer(
            captions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=50
        )
        
        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model(
            images=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx + 1}:")
        print(f"  Image batch shape: {pixel_values.shape}")
        print(f"  Text batch shape: {input_ids.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Captions: {captions}")
    
    # Save adapter weights
    print(f"\\n=== Saving/Loading Adapters ===")
    
    adapter_save_path = "my_trained_adapters.pt"
    model.save_adapters(adapter_save_path)
    
    # Create new model and load adapters
    model2 = RTVTLM(
        clip_model_name="openai/clip-vit-base-patch32",
        qwen_model_name="Qwen/Qwen2.5-0.5B-Instruct", 
        insert_positions=[2, 5, 8, 11, 14, 17, 20],
        device=device
    )
    
    model2.load_adapters(adapter_save_path)
    print(f"Successfully transferred adapters to new model instance!")
    
    print(f"\\n=== Complete! ===")
    print("The RTVTLM model is ready for:")
    print("1. Fine-tuning adapters on vision-language tasks")
    print("2. Image captioning and VQA")
    print("3. Multi-modal reasoning with frozen base models")
    
if __name__ == "__main__":
    example_usage()