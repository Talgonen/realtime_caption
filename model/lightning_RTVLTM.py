"""
PyTorch Lightning Module for Hybrid Vision-Language Model
"""

import torch
import pytorch_lightning as pl
from typing import Optional, Dict, Any

from transformers import AutoTokenizer, CLIPProcessor

from .RTVTLM import RTVTLMConfig, RTVTLMForCausalLM


class LightningRTVTLM(pl.LightningModule):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        qwen_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        insert_positions: Optional[list] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_generation_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        optimizer: str = "adamw",
        lr_scheduler: str = "cosine",
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize model (Lightning handles device placement automatically)
        self.model = RTVTLMForCausalLM.from_pretrained(
            qwen_model_name,
            clip_model_name,
            config=RTVTLMConfig(adapter_insert_positions=insert_positions),
            local_files_only=True,
        )

        for name, param in self.model.named_parameters():
            if "model.adapters" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.language_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, local_files_only=True)
        self.vision_processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True, use_fast=True)
        
        # Store training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_generation_length = max_generation_length
        self.temperature = temperature
        self.top_p = top_p
        self.optimizer_name = optimizer
        self.lr_scheduler_name = lr_scheduler
        self.verbose = verbose
    
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, texts = batch
        loss = self.loss_fn(images, texts, batch_idx)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, texts = batch
        loss = self.loss_fn(images, texts, batch_idx)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Generate sample captions on first batch
        if self.verbose and batch_idx == 0:
            self._generate_samples(images[:2], texts[:2])
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        images, texts = batch
        loss = self.loss_fn(images, texts, batch_idx)

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Generate sample captions on first batch
        if self.verbose and batch_idx == 0:
            self._generate_samples(images[:2], texts[:2])
        
        return loss
    
    def loss_fn(self, images, texts, _):
        inputs = self.language_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.hparams.get('max_seq_length', 512)
        ).to(self.device)
        
        # Process images and move to device
        pixel_values = self.vision_processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        vision_features = self.model.encode_vision(pixel_values)
        
        # Forward pass
        outputs = self(
            vision_features=vision_features,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=inputs.input_ids
        )
        
        return outputs.loss
    
    def _generate_samples(self, images, texts):
        """Generate sample captions for logging."""
        with torch.no_grad():
            for i, (image, text) in enumerate(zip(images, texts)):
                generated_text = self.model.generate(
                    pixel_values=self.vision_processor(images=image, return_tensors="pt").pixel_values.to(self.device),
                    max_length=self.max_generation_length,
                    temperature=self.temperature,
                    top_p=self.top_p
                )

                generated_text = self.language_tokenizer.decode(generated_text[0], skip_special_tokens=True)
                
                print(f"\n{'='*60}")
                print(f"Sample {i+1}:")
                print(f"Ground truth: {text}")
                print(f"Generated: {generated_text}")
                print('='*60)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get trainable parameters only
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        # Create optimizer
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Create learning rate scheduler
        if self.lr_scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.learning_rate * 0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        elif self.lr_scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.trainer.estimated_stepping_batches
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            # No scheduler
            return optimizer
    
    def on_train_start(self):
        """Called at the beginning of training."""
        print("\n" + "="*60)
        print("Model Parameter Statistics:")
        print("="*60)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Frozen Parameters: {frozen_params:,}")
        print(f"Trainable Percentage: {100 * trainable_params / total_params:.2f}%")
        
        print("="*60 + "\n")
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint."""
        # Add any custom state you want to save
        checkpoint['model_config'] = {
            'clip_model_name': self.hparams.clip_model_name,
            'qwen_model_name': self.hparams.qwen_model_name,
            'insert_positions': self.hparams.insert_positions
        }
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        # Restore any custom state
        if 'model_config' in checkpoint:
            print(f"Loading model config: {checkpoint['model_config']}")
