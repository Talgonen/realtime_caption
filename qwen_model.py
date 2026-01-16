from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

class VLModel:
    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", device=None):
        """
        Initialize the LLaVA OneVision model from Hugging Face.
        
        Args:
            model_name: The model identifier on Hugging Face
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {self.device}")
        
        # Load the model directly to GPU to avoid RAM overhead
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        print("Model loaded successfully!")
    
    def generate_caption(self, image):
        """
        Generate a caption for the given image.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            Generated caption as a string
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Prepare the prompt for captioning
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image with up to 7 words."},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        # Decode the output
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "assistant" in caption.lower():
            caption = caption.split("assistant")[-1].strip()
        
        return caption
