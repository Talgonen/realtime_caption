# Real-Time Vision to Language Model (RTVTLM)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6.0](https://img.shields.io/badge/pytorch-2.6.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid vision-language model combining CLIP and Qwen2.5 for real-time image and video understanding. This project aims to create an intelligent background assistant that continuously monitors and understands visual context.

## 🎯 Project Vision

This is a **multi-stage project** building towards an intelligent daily assistant:

### Stage 1: ✅ **CLIP-Qwen Adapter Training** (COMPLETE)
Adapt Qwen2.5 with CLIP to support insertion of image data in the middle of the model, enabling:
- Parallel processing of vision encoder and early language layers
- Reduced latency for real-time applications
- Efficient training with <1% trainable parameters

### Stage 2: 🔄 **Video Understanding** (IN PROGRESS)
Enable the model to generate descriptions for video streams with:
- Continuous caption generation as scenes change
- Attention-aware detail generation (when user dwells on an object, describe it in more detail)
- Temporal coherence across frames
- Real-time streaming inference

### Stage 3: 🔮 **Background Life Assistant** (PLANNED)
Create a background assistant that:
- Continuously monitors user's screen/environment
- Writes timestamped observations to memory
- Retrieves relevant information on user queries
- Acts as a "life logger" and personal knowledge base

---

## 🏗️ Architecture

### Current Model: RTVTLM (Real-Time Vision to Language Model)

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  CLIP Encoder   │  ← Frozen (ViT-B/32)
│  (Parallel)     │
└──────┬──────────┘
       │ Vision Features
       │
       ▼
┌──────────────────────────────────────────────────┐
│            Qwen2.5-0.5B-Instruct                 │
│                                                   │
│  Layer 0-14  ← Frozen (Run in parallel w/ CLIP) │
│  Layer 15    ← CrossAttentionAdapter (Trainable) │
│  Layer 16    ← Frozen                            │
│  Layer 17    ← CrossAttentionAdapter (Trainable) │
│  Layer 18    ← Frozen                            │
│  Layer 19    ← CrossAttentionAdapter (Trainable) │
│  Layer 20    ← Frozen                            │
│  Layer 21    ← CrossAttentionAdapter (Trainable) │
│  Layer 22    ← Frozen                            │
│  Layer 23    ← CrossAttentionAdapter (Trainable) │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Caption   │
└─────────────┘
```

### Key Innovation: Mid-Layer Vision Insertion

Unlike traditional VLMs that inject vision at the start, RTVTLM inserts vision features at **layers 15, 17, 19, 21, 23** using cross-attention adapters. This enables:

✅ **Parallel Processing**: Early Qwen layers (0-14) process text independently while CLIP encodes the image
✅ **Reduced Latency**: Critical for real-time applications
✅ **Parameter Efficiency**: Only ~2-5M trainable params (<1% of total 896M)
✅ **Strong Performance**: Leverages full pretrained knowledge of both models

---

## 📊 Current State (Stage 1)

### ✅ Implemented Features

- **Hybrid Architecture**: CLIP + Qwen2.5 with cross-attention adapters
- **Efficient Training**: Freeze base models, train only adapters (~2-5M params)
- **PyTorch Lightning**: Production-ready training infrastructure
- **COCO Dataset Integration**: Training on 75K COCO Karpathy samples
- **Mixed Precision Training**: FP16 for faster training
- **Real-Time GUI**: Tkinter overlay for live screen captioning
- **Checkpoint Management**: Save/load trained models
- **Inference Scripts**: Generate captions from images
- **Flexible Configuration**: YAML-based config system

### 📈 Training Status

| Metric | Value |
|--------|-------|
| **Dataset** | COCO Karpathy (yerevann) |
| **Train Samples** | 75,000 |
| **Validation Samples** | 20 |
| **Batch Size** | 4 × 16 accumulation = 64 effective |
| **Learning Rate** | 2e-3 (constant) |
| **Max Epochs** | 1,000 |
| **Trainable Params** | ~2-5M (0.56% of total) |
| **Total Params** | ~896M (CLIP + Qwen) |
| **GPU Memory** | ~6GB (GTX 1660 compatible) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA 12.4+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 6GB+ GPU VRAM (for training)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd realtime_caption

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download pretrained models (first run will auto-download)
# Models cached in ~/.cache/huggingface/
```

### Usage

#### 1. Train the Model

```bash
# Train with default config (recommended for 6GB GPU)
python training/train.py

# Train with custom config
python training/train.py --config training/configs/small_config.yaml

# Override specific parameters
python training/train.py --batch_size 2 --learning_rate 1e-3
```

#### 2. Generate Captions from Images

```bash
# Use trained checkpoint
python generate_caption.py \
  --checkpoint training/checkpoints/last.ckpt \
  --image COCO.jpg \
  --prompt "Describe this image in detail"

# With sampling
python generate_caption.py \
  --checkpoint training/checkpoints/last.ckpt \
  --image test.png \
  --do_sample \
  --temperature 0.7 \
  --num_beams 3
```

#### 3. Real-Time Screen Captioning GUI

```bash
# Launch overlay application
python main.py

# Controls:
# - SPACE: Start/pause captioning
# - ESC: Quit application
```

---

## 📁 Project Structure

```
realtime_caption/
├── model/
│   ├── RTVTLM.py                 # Core model architecture
│   │   ├── RTVTLMConfig           # Model configuration
│   │   ├── RTVTLMModel            # Base model with adapters
│   │   └── RTVTLMForCausalLM      # Causal LM head + generation
│   └── lightning_RTVLTM.py       # PyTorch Lightning wrapper
│       ├── LightningRTVTLM        # Training/validation loops
│       └── Optimizer/LR configs   # Training utilities
│
├── training/
│   ├── train.py                  # Main training script
│   ├── configs/
│   │   ├── default_config.yaml   # Default training config
│   │   ├── small_config.yaml     # For limited GPU memory
│   │   └── large_config.yaml     # For high-end GPUs
│   └── utils.py                  # Training utilities
│
├── dataset/
│   └── COCO.py                   # COCO Karpathy dataset loader
│       ├── COCOKarpathyDataset           # Base dataset
│       └── COCOKarpathyDatasetWithTokenizer  # With pre-tokenization
│
├── main.py                       # Real-time GUI application
├── generate_caption.py           # Inference script
├── test.py                       # Testing utilities
└── pyproject.toml                # Dependencies (uv)
```

---

## 🧠 Model Details

### CrossAttentionAdapter

Each trainable adapter consists of:

```python
CrossAttentionAdapter(
    hidden_size=896,              # Qwen hidden size
    vision_hidden_size=768,       # CLIP hidden size
    num_heads=8
)

Components:
├── Multi-head Cross-Attention    # Query: text, Key/Value: vision
├── Layer Normalization
└── Gated Residual Connection     # Learnable gate (init=0)
```

**Why Gated Residuals?**
Starting with gate=0 allows the model to initially behave exactly like pretrained Qwen, then gradually learn to incorporate vision information without disrupting pretrained knowledge.

### Training Strategy

1. **Freeze Everything**: CLIP and Qwen2.5 completely frozen
2. **Train Only Adapters**: 5 adapters at specific layers (15, 17, 19, 21, 23)
3. **Large Effective Batch**: Small GPU batch (4) × accumulation (16) = 64 effective
4. **Constant LR**: 2e-3 without decay (adapters are small and fast to train)
5. **Mixed Precision**: FP16 for 2× speedup and 50% memory savings

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CLIP**: [OpenAI CLIP](https://github.com/openai/CLIP)
- **Qwen2.5**: [Alibaba Qwen](https://github.com/QwenLM/Qwen2.5)
- **COCO Dataset**: [COCO Karpathy Split](https://huggingface.co/datasets/yerevann/coco-karpathy)
- **PyTorch Lightning**: [Lightning AI](https://github.com/Lightning-AI/pytorch-lightning)

---

## 📚 Additional Resources

### Papers
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2407.10671)
- [Flamingo: Visual Language Model](https://arxiv.org/abs/2204.14198)

### Related Projects
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Visual instruction tuning
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) - BLIP-2 + Vicuna
- [BLIP-2](https://github.com/salesforce/LAVIS) - Bootstrapping language-image pre-training

---

**Status**: Stage 1 Complete ✅ | Stage 2 In Progress 🔄 | Stage 3 Planned 🔮

Last Updated: March 18, 2026
