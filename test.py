from torch.profiler import profile, ProfilerActivity
import torch
from PIL import Image
import time
from torch.profiler import ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPProcessor, CLIPModel

from model.RTVTLM import RTVTLMConfig, RTVTLMForCausalLM

def check_image_model(image_path):
    """
    Check if an image matches the description "an image from a facebook page"
    
    Args:
        image_path: Path to the image file
    
    Returns:
        float: Probability score (0-1) that the image is from a facebook page
    """
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    
    # Warmup
    print("Warming up...")
    dummy_image = Image.new('RGB', (224, 224), color='white')
    dummy_inputs = processor(text=["test"], images=dummy_image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        _ = model(**dummy_inputs)
    
    # Load and preprocess the image
    image = Image.open(image_path)
    
    # Prepare text descriptions
    text_descriptions = [
        "an image from a facebook page",
        "a regular photo"
    ]
    
    # Process inputs
    inputs = processor(text=text_descriptions, images=[image]*32, return_tensors="pt", padding=True).to(device)
    
    # Calculate features
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        print(time.time() - start_time, "seconds to encode image")
        
        # Get the image-text similarity scores
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    facebook_probability = probs[0][0].item()
    
    print(f"Probability that the image is from a Facebook page: {facebook_probability:.2%}")
    print(f"Probability that it's a regular photo: {probs[0][1].item():.2%}")
    
    return facebook_probability

def check_text_model(prompt):
    """
    Generate text using Qwen2.5-0.5B model
    
    Args:
        prompt: Text prompt for generation
    
    Returns:
        str: Generated text
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Use -Instruct version
    language_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    print("Compiling model...")
    model = torch.compile(language_model, mode="max-autotune", fullgraph=True)
    
    # Warmup (IMPORTANT - first compile run is slow)
    print("Warming up...")
    dummy = tokenizer("test", return_tensors="pt").to(device)
    _ = model.generate(**dummy, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    print("Generating...")
    start_time = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,          # Record tensor shapes
            profile_memory=True,          # Track memory usage
            with_stack=True,              # Include Python call stack
            with_flops=True,              # Estimate FLOPs
            with_modules=True             # Track nn.Module hierarchy
            ) as prof:
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    prof.export_chrome_trace("trace.json")
    generation_time = time.time() - start_time
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"{generation_time:.4f} seconds to generate text")
    print(f"\nGenerated text:\n{generated_text}")
    
    return generated_text

def check_image_to_text_model(prompt):
    """
    Generate text using Qwen2.5-0.5B model
    
    Args:
        prompt: Text prompt for generation
    
    Returns:
        str: Generated text
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    language_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    vision_model_name = "openai/clip-vit-base-patch32"

    insert_positions = [20,21,22,23,24]  # Add adapters at these layer positions
    model = RTVTLMForCausalLM.from_pretrained(
        language_model_name,
        vision_model_name,
        config=RTVTLMConfig(adapter_insert_positions=insert_positions),
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(language_model_name, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(vision_model_name, local_files_only=True, use_fast=True,)
    
    print("Compiling model...")
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # Set up the image
    image = Image.open(image_path)
    image_input = processor(image, return_tensors="pt", padding=True).to(device)
    
    # Warmup (IMPORTANT - first compile run is slow)
    print("Warming up...")
    dummy = tokenizer("test", return_tensors="pt", padding=True).to(device)
    _ = model.generate(pixel_values=image_input.pixel_values, input_ids=dummy["input_ids"], attention_mask=dummy["attention_mask"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    
    # Tokenize input
    if len(prompt) > 0:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    else:
        inputs = tokenizer("A", return_tensors="pt", add_special_tokens=True).to(device)
        inputs['attention_mask'] = torch.zeros_like(inputs['input_ids'])
    image_input = processor(image, return_tensors="pt", padding=True).to(device)
    
    # Generate text
    print("Generating...")
    start_time = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,          # Record tensor shapes
            profile_memory=True,          # Track memory usage
            with_stack=True,              # Include Python call stack
            with_flops=True,              # Estimate FLOPs
            with_modules=True             # Track nn.Module hierarchy
            ) as prof:
        outputs = model.generate(
            pixel_values=image_input.pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
        )
    prof.export_chrome_trace("trace.json")
    generation_time = time.time() - start_time
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"{generation_time:.4f} seconds to generate text")
    print(f"\nGenerated text:\n{generated_text}")
    
    return generated_text


if __name__ == "__main__":
    # Example usage - replace with your image path
    image_path = "COCO_val2014_000000403013.jpg"
    # check_image_model(image_path)
    # check_text_model("Describe a beautiful sunset over the mountains.")
    check_image_to_text_model("A narrow ")
