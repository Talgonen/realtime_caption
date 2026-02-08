import base64
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

class VLModel:
    def __init__(self, 
                 model_path=r"D:\huggingface\models--Qwen--Qwen3-VL-2B-Instruct-GGUF\snapshots\52d6c8ffea26cc873ac5ad116f8631268d7eb503\Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                 mmproj_path=r"D:\huggingface\models--Qwen--Qwen3-VL-2B-Instruct-GGUF\snapshots\52d6c8ffea26cc873ac5ad116f8631268d7eb503\mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
                 n_gpu_layers=-1,
                 n_ctx=2048,
                 **kwargs):
        """
        Initialize the Qwen3-VL model using llama-cpp-python.
        
        Args:
            model_path: Path to the GGUF model file
            mmproj_path: Path to the multimodal projector GGUF file
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context size
        """
        print(f"Loading model from {model_path}")
        
        # Create the chat handler for vision models
        self.chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path)
        
        # Load the model
        self.model = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,
        )
        
        print("Model loaded successfully!")
    
    def _image_to_base64(self, image):
        """Convert PIL Image to base64 data URI."""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    
    def generate_caption(self, image):
        """
        Generate a caption for the given image.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            Generated caption as a string
        """
        image_uri = self._image_to_base64(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": "Describe this image with up to 7 words."}
                ]
            }
        ]
        
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0
        )
        
        caption = response["choices"][0]["message"]["content"]
        return caption.strip()
    
    def generate_captions_batch(self, images):
        """
        Generate captions for a batch of images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of generated captions as strings
        """
        if not images:
            return []
        
        # Process sequentially (llama-cpp processes one at a time)
        captions = []
        for image in images:
            caption = self.generate_caption(image)
            captions.append(caption)
        
        return captions
