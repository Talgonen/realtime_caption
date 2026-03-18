# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Stage 2: Video Understanding (In Progress)
- [ ] Video frame extraction pipeline
- [ ] Temporal modeling extensions
- [ ] Attention-aware generation
- [ ] Video dataset integration (MSR-VTT)

### Stage 3: Background Assistant (Planned)
- [ ] Background capture service
- [ ] Memory storage system
- [ ] Retrieval and query interface
- [ ] Privacy filters

## [0.1.0] - 2026-03-18

### Added - Stage 1 Complete ✅

#### Core Model
- Implemented `RTVTLMForCausalLM` with mid-layer vision injection
- CrossAttentionAdapter at layers [15, 17, 19, 21, 23]
- CLIP ViT-B/32 vision encoder integration
- Qwen2.5-0.5B-Instruct language model integration
- Gated residual connections for stable training

#### Training Infrastructure
- PyTorch Lightning wrapper (`LightningRTVTLM`)
- YAML-based configuration system
- COCO Karpathy dataset integration
- Mixed precision (FP16) training support
- Gradient accumulation for large effective batch sizes
- TensorBoard logging
- Model checkpointing with automatic best model selection

#### Applications
- Real-time screen captioning GUI (tkinter overlay)
- Inference script for single-image captioning
- Image slicing capability for detailed analysis

#### Configuration Presets
- `default_config.yaml` - Balanced config for 6GB GPU
- `small_config.yaml` - For limited memory scenarios
- `large_config.yaml` - For high-end GPUs

#### Training Details
- 75,000 COCO training samples
- 20 validation samples
- Batch size: 4 × 16 accumulation = 64 effective
- Learning rate: 2e-3 (constant)
- Only ~2-5M trainable parameters (<1% of total)

### Technical Details

#### Model Architecture
- Total parameters: ~896M (CLIP + Qwen)
- Trainable parameters: ~2-5M (0.56%)
- Vision insertion: Layers 15, 17, 19, 21, 23
- Cross-attention heads: 8
- Hidden size: 896 (Qwen)
- Vision hidden size: 768 (CLIP)

#### Performance
- Training speed: ~60-500 samples/sec (depending on GPU)
- Inference latency: ~30-500ms (depending on hardware)
- Memory usage: ~6-18GB (depending on batch size)

## [0.0.1] - Initial Commits

### 2026-03-18 - Training Branch Created
- **Commit 405c605**: Add training for alignment between qwen2.5 and CLIP
- **Commit 27fcea8**: Add CLIP to Qwen2 and setup basic training (no dataset yet)

### 2026-03-18 - Initial Implementation
- **Commit feb2132**: Initial commit: real-time screen captioning with Qwen VL
- Basic project structure
- Initial model architecture exploration

---

## Version History Summary

| Version | Stage | Status | Key Features |
|---------|-------|--------|--------------|
| **0.1.0** | Stage 1 | ✅ Complete | CLIP-Qwen adapters, COCO training, RT GUI |
| **0.2.0** | Stage 2 | 🔄 In Progress | Video understanding, attention tracking |
| **0.3.0** | Stage 3 | 🔮 Planned | Background assistant, memory system |

---

## Upgrade Notes

### From Initial Version to 0.1.0

- Model architecture changed to mid-layer vision injection
- Training now uses PyTorch Lightning (replaces manual training loops)
- Configuration moved from Python to YAML files
- Dataset switched from custom loader to COCO Karpathy

### Future Breaking Changes (0.2.0)

- Model will be extended to handle video inputs
- New temporal modeling components
- API changes for generation (streaming tokens)

---

## Contributors

- Initial development and architecture design
- COCO dataset integration
- PyTorch Lightning migration
- Documentation

---

Last Updated: March 18, 2026
