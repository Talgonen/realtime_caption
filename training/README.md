# Training Setup for RTVTLM

> **📘 Note**: For overall project vision, architecture, and roadmap, see the [main README](../README.md).
> This document focuses specifically on training the model.

## Overview

This folder contains the training infrastructure for RTVTLM (Real-Time Vision to Language Model), which combines:
- **Frozen CLIP** (vision encoder)
- **Frozen Qwen2.5-0.5B** (language model)
- **Trainable intermediate layers** inserted between transformer blocks

## 📁 Folder Structure

```
training/
├── models/
│   └── hybrid_model.py          # Hybrid model architecture
├── configs/
│   └── config.py                # Training configurations
├── checkpoints/                 # Saved model checkpoints
├── logs/                        # TensorBoard logs
├── train.py                     # Main training script
└── utils.py                     # Utility functions
```

## 🏗️ Model Architecture

The hybrid model consists of:

1. **CLIP Vision Encoder** (Frozen)
   - Encodes input images into feature vectors
   - Default: ViT-B/32

2. **Vision-to-Text Projection** (Trainable)
   - Projects CLIP features to Qwen's hidden dimension
   - Linear layer: `clip_dim → qwen_hidden_size`

3. **Qwen Language Model** (Frozen) with Intermediate Layers (Trainable)
   - Qwen2.5-0.5B base model
   - Custom intermediate layers inserted between transformer blocks
   - Each intermediate layer contains:
     - Feed-forward network
     - GELU activation
     - Residual connection
     - Layer normalization

## 🚀 Quick Start

### 1. Install Dependencies

Make sure you have the required packages:
```bash
uv add transformers torch torchvision "git+https://github.com/openai/CLIP.git" pillow
```

### 2. Test Model Initialization

```bash
cd training/models
python hybrid_model.py
```

This will:
- Load CLIP and Qwen models
- Insert intermediate layers at specified positions
- Print parameter statistics

### 3. View Configuration

```bash
cd training/configs
python config.py
```

### 4. Start Training

```bash
cd training
python train.py
```

**Note**: The current training script uses a dummy dataset. You'll need to replace `DummyDataset` in `train.py` with your actual dataset.

## ⚙️ Configuration

Edit `training/configs/config.py` to customize:

### Model Configuration
```python
model_config = ModelConfig(
    clip_model_name="ViT-B/32",           # CLIP variant
    qwen_model_name="Qwen/Qwen2.5-0.5B",  # Qwen variant
    insert_positions=[2, 4, 6],            # Where to insert layers
    freeze_clip=True,
    freeze_qwen=True
)
```

### Training Configuration
```python
training_config = TrainingConfig(
    batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
    gradient_accumulation_steps=1,
    use_fp16=True
)
```

## 🎯 Key Features

### Intermediate Layer Insertion

The model allows you to insert trainable layers between any transformer blocks:

```python
# Insert layers after blocks 2, 4, and 6
model = HybridVisionLanguageModel(
    insert_positions=[2, 4, 6]
)
```

### Only Train What You Need

- CLIP parameters: **Frozen** ❄️
- Qwen parameters: **Frozen** ❄️
- Vision projection: **Trainable** 🔥
- Intermediate layers: **Trainable** 🔥

This approach:
- Reduces memory requirements
- Speeds up training
- Prevents catastrophic forgetting of pretrained knowledge

### Forward Pass

```python
outputs = model(
    images=image_tensor,           # [B, 3, H, W]
    input_ids=token_ids,           # [B, seq_len]
    attention_mask=attention_mask, # [B, seq_len]
    labels=labels                  # [B, seq_len] (optional)
)

loss = outputs.loss
```

### Generation

```python
caption = model.generate(
    images=image_tensor,
    prompt_text="A photo of",
    max_length=100,
    temperature=0.7
)
```

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir training/logs
```

View:
- Training loss
- Validation loss
- Learning rate
- Generated samples

### Weights & Biases (Optional)

Enable in `config.py`:
```python
logging_config = LoggingConfig(
    use_wandb=True,
    wandb_project="vision-language-model"
)
```

## 💾 Checkpoints

Checkpoints are automatically saved to `training/checkpoints/`:
- `checkpoint_step_N.pt` - Regular checkpoints
- `best_model.pt` - Best model based on validation loss

Load a checkpoint:
```python
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🔧 Customization

### Adding Custom Intermediate Layers

Modify `IntermediateLayer` in `hybrid_model.py`:

```python
class IntermediateLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # Add your custom architecture here
        self.custom_layer = nn.Linear(hidden_size, hidden_size)
```

### Using Your Own Dataset

Replace `DummyDataset` in `train.py`:

```python
class MyDataset(Dataset):
    def __init__(self, data_path):
        # Load your data
        self.images = ...
        self.captions = ...
    
    def __getitem__(self, idx):
        image = load_image(self.images[idx])
        caption = self.captions[idx]
        return image, caption
```

## 📈 Expected Results

With default settings:
- **Total Parameters**: ~896M (CLIP + Qwen)
- **Trainable Parameters**: ~2-5M (projection + intermediate layers)
- **Trainable Ratio**: < 1%

This efficient approach allows:
- Fast training on consumer GPUs
- Low memory footprint
- Good generalization

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Use fewer `insert_positions`

### Slow Training
- Increase `num_workers` for data loading
- Enable `use_fp16` for mixed precision
- Use `pin_memory=True`

### Model Not Learning
- Check learning rate (try 1e-4 to 5e-4)
- Verify only intended layers are trainable
- Ensure data preprocessing is correct

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2407.10671)
- [Vision-Language Models Survey](https://arxiv.org/abs/2304.00685)
