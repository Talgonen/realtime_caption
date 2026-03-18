# Contributing to RTVTLM

Thank you for your interest in contributing to the Real-Time Vision-Language Model project! This document provides guidelines and information for contributors.

## 🎯 Project Status

**Current Stage**: Stage 1 Complete ✅, Stage 2 In Progress 🔄

See [ROADMAP.md](ROADMAP.md) for detailed development plans.

---

## 🤝 How to Contribute

### Areas We Need Help With

#### Stage 2: Video Understanding (Current Priority)
- [ ] **Video dataset integration** (MSR-VTT, YouCook2)
- [ ] **Temporal modeling** (extending RTVTLM for video)
- [ ] **Attention tracking** (mouse/gaze-based ROI detection)
- [ ] **Object detection integration** (YOLO/DINO)
- [ ] **Streaming inference pipeline**
- [ ] **Performance optimization**

#### Stage 3: Background Assistant (Future)
- [ ] **Memory storage system** (JSONL + vector DB)
- [ ] **Privacy filters** (PII detection)
- [ ] **Retrieval system** (semantic search)
- [ ] **Query interface** (CLI/GUI)

#### General Improvements
- [ ] **Documentation** (tutorials, examples, docstrings)
- [ ] **Testing** (unit tests, integration tests)
- [ ] **Evaluation metrics** (BLEU, CIDEr, SPICE)
- [ ] **Model optimization** (quantization, pruning)
- [ ] **Dataset expansion** (more training data)
- [ ] **Bug fixes** (see Issues)

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/realtime_caption.git
cd realtime_caption
```

### 2. Set Up Development Environment

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install development dependencies
uv add --dev pytest black ruff mypy
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

---

## 📝 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

```bash
# Format code with black
black .

# Lint with ruff
ruff check .

# Type check with mypy
mypy model/ training/ dataset/
```

**Key conventions**:
- Line length: 100 characters (not 80)
- Use type hints for function signatures
- Docstrings: Google style
- Imports: grouped (stdlib, third-party, local)

### Example Function

```python
from typing import Optional, List
import torch
from transformers import AutoTokenizer


def process_captions(
    captions: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 77,
    padding: str = "max_length"
) -> torch.Tensor:
    """
    Process a batch of captions into tokenized tensors.

    Args:
        captions: List of caption strings to tokenize
        tokenizer: Hugging Face tokenizer instance
        max_length: Maximum sequence length for padding/truncation
        padding: Padding strategy ('max_length', 'longest', 'do_not_pad')

    Returns:
        Tokenized caption tensor of shape (batch_size, max_length)

    Example:
        >>> captions = ["A dog", "A cat on a mat"]
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokens = process_captions(captions, tokenizer)
        >>> tokens.shape
        torch.Size([2, 77])
    """
    tokenized = tokenizer(
        captions,
        max_length=max_length,
        padding=padding,
        truncation=True,
        return_tensors="pt"
    )
    return tokenized["input_ids"]
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=model --cov=training --cov=dataset

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_model.py
import pytest
import torch
from model.RTVTLM import RTVTLMConfig, RTVTLMForCausalLM


def test_model_initialization():
    """Test that model initializes correctly."""
    config = RTVTLMConfig(
        adapter_insert_positions=[2, 4, 6]
    )
    model = RTVTLMForCausalLM(config)

    assert len(model.model.adapters) == 3
    assert "2" in model.model.adapters


@pytest.mark.slow
def test_model_forward():
    """Test forward pass (slow test)."""
    # ... test implementation
```

---

## 📊 Benchmarking

Before submitting performance-related PRs, please benchmark:

```bash
# Training speed
python benchmark/train_speed.py --batch_size 4 --num_steps 100

# Inference latency
python benchmark/inference_latency.py --model checkpoint.ckpt --num_samples 100

# Memory usage
python benchmark/memory_usage.py --batch_size 4
```

Include results in your PR description.

---

## 📚 Documentation

### Docstrings

All public functions, classes, and modules must have docstrings:

```python
class VideoRTVTLM(RTVTLMForCausalLM):
    """
    Extension of RTVTLM for video understanding.

    This model processes sequences of video frames with temporal modeling
    to generate coherent video captions.

    Attributes:
        temporal_embeddings: Learnable positional embeddings for frames
        frame_aggregator: Temporal attention module

    Example:
        >>> model = VideoRTVTLM.from_pretrained("qwen-model", "clip-model")
        >>> frames = [frame1, frame2, frame3]  # List of PIL Images
        >>> caption = model.generate_video_caption(frames)
    """
```

### README Updates

If your contribution adds new features:
1. Update main [README.md](README.md)
2. Update [ROADMAP.md](ROADMAP.md) if it affects future plans
3. Add entry to [CHANGELOG.md](CHANGELOG.md)

---

## 🔄 Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines (black, ruff)
- [ ] Tests pass (`pytest`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive

### 2. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Features
git commit -m "feat: add temporal attention for video modeling"
git commit -m "feat(dataset): integrate MSR-VTT video dataset"

# Bug fixes
git commit -m "fix: resolve memory leak in frame buffer"
git commit -m "fix(training): correct gradient accumulation logic"

# Documentation
git commit -m "docs: add video processing tutorial"
git commit -m "docs(api): improve docstrings for RTVTLM"

# Performance
git commit -m "perf: optimize CLIP batch encoding"

# Refactor
git commit -m "refactor: simplify attention adapter code"

# Tests
git commit -m "test: add unit tests for video dataset"
```

### 3. PR Template

When opening a PR, use this template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring

## Related Issues
Fixes #123

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Benchmarks (if applicable)
- Training speed: X samples/sec
- Inference latency: X ms
- Memory usage: X GB

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG updated
```

### 4. Review Process

1. **Automated checks** must pass (linting, tests)
2. **Code review** by maintainers
3. **Feedback addressed** (if any)
4. **Merge** when approved

---

## 🐛 Reporting Bugs

### Before Reporting

1. Check [existing issues](https://github.com/yourrepo/issues)
2. Try latest version (`git pull origin training`)
3. Minimal reproducible example

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## To Reproduce
1. Step 1
2. Step 2
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.12.1]
- PyTorch: [e.g., 2.6.0]
- GPU: [e.g., RTX 3060]
- CUDA: [e.g., 12.4]

## Logs/Screenshots
```python
# Error traceback
```

## Additional Context
Any other relevant information
```

---

## 💡 Feature Requests

We welcome feature suggestions! Please include:

1. **Use case**: Why is this feature needed?
2. **Proposal**: How should it work?
3. **Alternatives**: What other options did you consider?
4. **Willing to implement**: Can you contribute code?

---

## 🏷️ Labeling Issues

| Label | Description |
|-------|-------------|
| `stage-2` | Video understanding related |
| `stage-3` | Background assistant related |
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `performance` | Speed/memory optimization |
| `testing` | Test coverage improvements |

---

## 🎓 Learning Resources

### Understanding the Architecture

1. Read [README.md](README.md) - Project overview
2. Read [training/README.md](training/README.md) - Training details
3. Read [ROADMAP.md](ROADMAP.md) - Future plans
4. Study `model/RTVTLM.py` - Core architecture

### Key Papers

- **CLIP**: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- **Qwen2.5**: [Qwen Technical Report](https://arxiv.org/abs/2407.10671)
- **Flamingo**: [Visual Language Model](https://arxiv.org/abs/2204.14198)
- **LLaVA**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

### PyTorch Lightning

- [Official Docs](https://lightning.ai/docs/pytorch/stable/)
- [Tutorial: Fine-tuning](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience, education
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Accepting constructive criticism gracefully
- Focusing on what is best for the community

**Unacceptable behaviors:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations can be reported to project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

## 📞 Getting Help

### Community Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcase
- **Email**: maintainer@example.com (for sensitive issues)

### Office Hours (Optional)

We may schedule virtual office hours for real-time help. Check Discussions for announcements.

---

## 🏆 Recognition

Contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Mentioned in release notes
- Credited in papers/publications (if applicable)

Significant contributions may earn:
- Collaborator status (write access)
- Co-authorship on research papers
- Speaking opportunities at presentations

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers this project.

---

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

If you have questions about contributing, feel free to open an issue with the `question` label.

---

Last Updated: March 18, 2026
