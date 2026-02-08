# RTVTLM Model - Fixed Implementation

## Overview

The RTVTLM (Real-Time Vision-Text Language Model) is a hybrid model that combines:
- **Frozen CLIP vision encoder** (openai/clip-vit-base-patch32) 
- **Frozen Qwen2.5-0.5B language model**
- **Trainable cross-attention adapters** inserted between transformer layers

## Key Features

### ✅ Fixed Issues
1. **Proper adapter initialization** - Adapters are correctly created and positioned
2. **Dtype consistency** - All components use float16 on GPU for memory efficiency
3. **Device placement** - Proper CUDA/CPU handling throughout
4. **Forward pass** - Uses language model's built-in forward method with hidden states
5. **Parameter freezing** - Base models frozen, adapters trainable
6. **Save/Load functionality** - Can save and load only adapter weights

### Model Architecture
```
Input Image → CLIP Vision Encoder (frozen) → Vision Features
Input Text → Qwen2.5 Embeddings → Transformer Layers with Adapters → Output Logits

Adapters are inserted at specified positions:
- Layer 3: CrossAttentionAdapter(896, 768)  
- Layer 6: CrossAttentionAdapter(896, 768)
- Layer 9: CrossAttentionAdapter(896, 768)  
- Layer 12: CrossAttentionAdapter(896, 768)
```

### Parameter Efficiency
- **Total parameters**: ~593M
- **Trainable parameters**: ~12M (2.01%)
- **Frozen parameters**: ~581M (97.99%)

## Usage

### Basic Initialization
```python
from training.models.rt_vlm import RTVTLM

model = RTVTLM(
    clip_model_name="openai/clip-vit-base-patch32",
    qwen_model_name="Qwen/Qwen2.5-0.5B-Instruct", 
    insert_positions=[3, 6, 9, 12],  # Adapter positions
    device="cuda"
)
```

### Training
```python
# Only adapter parameters are trainable
optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)

# Forward pass with loss
outputs = model(
    images=pixel_values,
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

loss = outputs['loss']
loss.backward()
optimizer.step()
```

### Inference
```python
model.eval()
outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
logits = outputs['logits']
```

### Save/Load Adapters
```python
# Save only adapter weights (lightweight)
model.save_adapters("my_adapters.pt")

# Load into new model instance
new_model = RTVTLM(...)
new_model.load_adapters("my_adapters.pt")
```

## Files Created/Modified

1. **`training/models/rt_vlm.py`** - Main model implementation
2. **`test_model.py`** - Comprehensive test suite  
3. **`example_usage.py`** - Usage examples and training demo

## Testing

Run the test suite to verify everything works:
```bash
uv run python test_model.py
```

Expected output:
- ✅ Forward pass successful
- ✅ Generation successful  
- ✅ Adapter save/load successful
- Parameter counts and model info

## Benefits

1. **Memory efficient** - Only trains 2% of parameters
2. **Flexible architecture** - Can specify any adapter positions
3. **Pre-trained weights** - Leverages CLIP and Qwen2.5 capabilities
4. **Easy deployment** - Save/load only small adapter files
5. **Modular design** - Can swap different base models easily

The model is now ready for fine-tuning on vision-language tasks while preserving the capabilities of both the vision and language components!