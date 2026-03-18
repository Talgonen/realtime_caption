"""
Script to generate captions from a trained checkpoint
"""

import torch
from PIL import Image
import argparse
from pathlib import Path

from model.lightning_RTVLTM import LightningRTVTLM


def generate_caption(
    checkpoint_path: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = False,
    num_beams: int = 1,
):
    """
    Generate a caption for an image using a trained model checkpoint.
    
    Args:
        checkpoint_path: Path to the Lightning checkpoint file
        image_path: Path to the input image
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (vs greedy/beam search)
        num_beams: Number of beams for beam search
    """
    # Load the model from checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = LightningRTVTLM.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load and process image
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Process image with vision processor
    pixel_values = model.vision_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Tokenize prompt
    input_ids = model.language_tokenizer(model.language_tokenizer.bos_token + prompt,
                                         return_tensors="pt").input_ids.to(device)
    
    # Generate caption
    print(f"\nGenerating caption with prompt: '{prompt}'")
    print("-" * 60)
    
    with torch.no_grad():
        generated_ids = model.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.language_tokenizer.pad_token_id,
            eos_token_id=model.language_tokenizer.eos_token_id,
            bos_token_id=model.language_tokenizer.bos_token_id,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            num_beams=num_beams,
        )
    
    # Decode generated text
    generated_text = model.language_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Generated Caption: {generated_text}")
    print("-" * 60)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate captions from images using trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"training\checkpoints\last.ckpt",
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test.png",
        help="Path to the input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail, including objects, their attributes, spatial relationships and background. Use no less than 60 words",
        help="Text prompt to start generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Generate caption
    generate_caption(
        checkpoint_path=str(checkpoint_path),
        image_path=str(image_path),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    main()
