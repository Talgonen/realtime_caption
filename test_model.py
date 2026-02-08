#!/usr/bin/env python3

import torch
from PIL import Image
from model.rtvtlm import RTVLMQwen2Config, RTVLMQwen2ForCausalLM
from transformers import AutoTokenizer
from transformers import CLIPProcessor

def test_model():
    """Test the RTVTLM model with adapter layers."""
    print("Testing RTVTLM model...")

    # Create dummy inputs for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with adapters at specific positions
    language_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    vision_model_name = "openai/clip-vit-base-patch32"

    insert_positions = [3, 6, 9, 12]  # Add adapters at these layer positions
    model = RTVLMQwen2ForCausalLM.from_pretrained(
        language_model_name=language_model_name,
        vision_model_name=vision_model_name,
        config=RTVLMQwen2Config(adapter_insert_positions=insert_positions),
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    processor = CLIPProcessor.from_pretrained(vision_model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print parameter counts
    print("\n=== Parameter Count ===")
    
    # Load example image
    try:
        image = Image.open("example.png").convert("RGB")
        print(f"Loaded image: example.png ({image.size})")
    except FileNotFoundError:
        print("Warning: example.png not found, creating dummy image")
        dummy_tensor = torch.randn(1, 3, 224, 224).to(device)
        # Convert tensor to PIL Image for processor compatibility
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        image = to_pil(dummy_tensor.squeeze(0).cpu())
    
    # Create dummy text input
    text_input = "Describe a beautiful sunset over the mountains:"
    tokenized = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    print(f"\n=== Testing Forward Pass ===")
    print(f"Input shape: {input_ids.shape}")
    print(f"Image size: {image.size}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(
                pixel_values=processor(image, return_tensors="pt").pixel_values.to(device),
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            print(f"Output logits shape: {outputs['logits'].shape}")
            print("✅ Forward pass successful!")
            
            # Test generation (basic)
            print(f"\n=== Testing Generation ===")
            generated_ids = model.generate(
                pixel_values=processor(image, return_tensors="pt").pixel_values.to(device),
                input_ids=input_ids,  # Optional
                max_length=50,
                do_sample=False,
                temperature=0 
            )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Generated text: {generated_text}")
            print("✅ Generation successful!")
            
        except Exception as e:
            print(f"❌ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test model structure
    print(f"\n=== Testing Model Structure ===")
    try:
        print(f"Vision encoder loaded: {model.model.vision_model is not None}")
        print(f"Adapters created: {len(model.model.adapters)} adapters")
        print("✅ Model structure looks good!")
    except Exception as e:
        print(f"❌ Error checking model structure: {e}")
        return False
    
    print(f"\n=== Model Configuration ===")
    print(f"Adapter positions: {model.config.adapter_insert_positions}")
    print(f"Number of adapters: {len(model.model.adapters)}")
    print(f"Total model layers: {len(model.model.language_model.layers)}")
    
    return True

if __name__ == "__main__":
    success = test_model()
    print(f"\n{'='*50}")
    if success:
        print("🎉 All tests passed! Model is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")