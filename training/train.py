"""
Training script for Hybrid Vision-Language Model using PyTorch Lightning
"""

import os
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.lightning_RTVLTM import LightningRTVTLM
from training.configs.config import load_config, load_default_config


class DummyDataset(Dataset):
    """Dummy dataset for testing. Replace with your actual dataset."""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Replace this with actual image loading and preprocessing
        image = torch.randn(3, 224, 224)
        text = "This is a sample caption for the image."
        return image, text


def create_dataloader(config, split: str) -> DataLoader:
    """Create dataloader for the specified split."""
    # TODO: Replace with actual dataset implementation
    dataset = DummyDataset(size=100 if split == "train" else 20)
    
    num_workers = config['data'].get('num_workers', 4)
    
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=True if num_workers > 0 else False
    )


def setup_callbacks(config):
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['output_dir'],
        filename='checkpoint-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=config['logging'].get('save_total_limit', 3),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    return callbacks


def setup_loggers(config):
    """Setup PyTorch Lightning loggers."""
    loggers = []
    
    # TensorBoard logger
    if config['logging'].get('use_tensorboard', True):
        tb_logger = TensorBoardLogger(
            save_dir=config['logging']['log_dir'],
            name="hybrid_model"
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    if config['logging'].get('use_wandb', False):
        wandb_logger = WandbLogger(
            project=config['logging'].get('wandb_project', 'vision-language-model'),
            name=config['logging'].get('wandb_run_name'),
            save_dir=config['logging']['log_dir']
        )
        loggers.append(wandb_logger)
    
    return loggers if loggers else True  # True uses default logger


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Hybrid Vision-Language Model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file (default: use default_config.yaml)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Override number of epochs')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = load_default_config()
        print("Using default configuration")
    
    # Override with command-line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    # Set seed for reproducibility
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    # Create output directories
    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Initialize model
    print("\nInitializing Lightning Module...")
    model = LightningRTVTLM(
        clip_model_name=config['model']['clip_model_name'],
        qwen_model_name=config['model']['qwen_model_name'],
        insert_positions=config['model'].get('insert_positions', [2, 4, 6]),
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
        warmup_steps=config['training'].get('warmup_steps', 500),
        max_generation_length=config['training'].get('max_generation_length', 100),
        temperature=config['training'].get('temperature', 0.7),
        top_p=config['training'].get('top_p', 0.9),
        optimizer=config['training'].get('optimizer', 'adamw'),
        lr_scheduler=config['training'].get('lr_scheduler', 'constant'),
        max_seq_length=config['data'].get('max_seq_length', 512)
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(config, "train")
    val_loader = create_dataloader(config, "val")
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(config)
    loggers = setup_loggers(config)
    
    # Calculate eval_interval
    eval_interval = config['logging'].get('eval_interval', 500)
    if eval_interval < 1.0:
        val_check_interval = eval_interval
    else:
        val_check_interval = int(eval_interval)
    
    # Initialize Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,  # Use single GPU to avoid multi-GPU issues
        precision="16-mixed" if config['training'].get('use_fp16', True) else 32,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config['logging'].get('log_interval', 10),
        val_check_interval=val_check_interval,
        accumulate_grad_batches=config['training'].get('gradient_accumulation_steps', 1),
        gradient_clip_val=config['training'].get('max_grad_norm', 1.0),
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=True,
    )
    
    # Print training info
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Max epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Optimizer: {config['training'].get('optimizer', 'adamw')}")
    print(f"LR Scheduler: {config['training'].get('lr_scheduler', 'cosine')}")
    print(f"Gradient accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    print(f"Mixed precision: {config['training'].get('use_fp16', True)}")
    print(f"Checkpoint dir: {config['logging']['output_dir']}")
    
    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("="*60 + "\n")
    
    # Start training with error handling
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        print("="*60)
    except RuntimeError as e:
        print("\n" + "="*60)
        print("ERROR: Training failed!")
        print("="*60)
        print(f"Error message: {str(e)}")
        if "out of memory" in str(e).lower():
            print("\nSUGGESTIONS:")
            print("1. Reduce batch_size in config (try batch_size: 1)")
            print("2. Increase gradient_accumulation_steps")
            print("3. Disable mixed precision (use_fp16: false)")
            print("4. Use smaller model")
        print("="*60)
        raise
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        raise


if __name__ == "__main__":
    main()
