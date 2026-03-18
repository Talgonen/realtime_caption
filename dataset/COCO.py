"""
COCO Karpathy Dataset
Based on yerevann/coco-karpathy from Hugging Face
Provides input images and output captions for image captioning tasks
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from typing import Optional, Callable, Dict, Any
import io
import requests
from pathlib import Path


class COCOKarpathyDataset(Dataset):
    """
    PyTorch Dataset wrapper for yerevann/coco-karpathy dataset
    
    Args:
        split (str): Dataset split - 'train', 'val', or 'test'
        transform (Callable, optional): Image transform/preprocessing function
        max_samples (int, optional): Maximum number of samples to load (for debugging)
        cache_dir (str, optional): Directory to cache downloaded images
        download_images (bool): If True, download images from URLs. Default: True
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        download_images: bool = True
    ):
        assert split in ['train', 'validation', 'test'], f"Split must be 'train', 'validation', or 'test', got {split}"
        
        self.split = split
        self.transform = transform
        self.download_images = download_images
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset from Hugging Face
        print(f"Loading yerevann/coco-karpathy dataset (split: {split})...")
        self.dataset = load_dataset('yerevann/coco-karpathy', split=split)
        
        # Limit samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples from {split} split")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict: Dictionary containing:
                - 'image': PIL Image or transformed tensor
                - 'caption': str or list of str (captions)
                - 'image_id': int (COCO image ID)
                - 'filename': str (image filename)
        """
        item = self.dataset[idx]
        
        # Get image from URL or cache
        image = self._load_image(item)
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Get captions
        captions = item.get('sentences', [])
        
        # Get image ID
        image_id = item.get('cocoid', item.get('imgid', idx))
        
        return {
            'image': image,
            'caption': captions,
            'image_id': image_id,
            'filename': item.get('filename', ''),
            'url': item.get('url', '')
        }
    
    def _load_image(self, item: Dict[str, Any]) -> Image.Image:
        """Load image from cache or download from URL"""
        filename = item.get('filename', f"image_{item.get('cocoid', 0)}.jpg")
        
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / filename
            if cache_path.exists():
                try:
                    return Image.open(cache_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading cached image {cache_path}: {e}")
        
        # Download from URL
        if self.download_images:
            url = item.get('url', '')
            if url:
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    
                    # Save to cache if cache_dir is specified
                    if self.cache_dir:
                        cache_path = self.cache_dir / filename
                        image.save(cache_path)
                    
                    return image
                except Exception as e:
                    print(f"Error downloading image from {url}: {e}")
                    # Return a blank image as fallback
                    return Image.new('RGB', (224, 224), color='gray')
        
        # Fallback: return blank image
        return Image.new('RGB', (224, 224), color='gray')
    
    def get_vocab_size(self) -> Optional[int]:
        """
        Returns vocabulary size if available in dataset info
        """
        return None  # Can be extended based on tokenizer


class COCOKarpathyDatasetWithTokenizer(COCOKarpathyDataset):
    """
    Extended COCO Dataset that tokenizes captions
    
    Args:
        split (str): Dataset split - 'train', 'val', or 'test'
        transform (Callable, optional): Image transform/preprocessing function
        tokenizer (Callable): Tokenizer function for captions
        max_length (int): Maximum caption length
        max_samples (int, optional): Maximum number of samples to load
        cache_dir (str, optional): Directory to cache downloaded images
        download_images (bool): If True, download images from URLs
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 77,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        download_images: bool = True
    ):
        super().__init__(
            split=split, 
            transform=transform, 
            max_samples=max_samples,
            cache_dir=cache_dir,
            download_images=download_images
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        
        # Tokenize caption if tokenizer is provided
        if self.tokenizer is not None:
            caption = item['caption']
            # Handle multiple captions - take the first one
            if isinstance(caption, list) and len(caption) > 0:
                caption = caption[0]
            
            # Tokenize
            tokenized = self.tokenizer(
                self.tokenizer.bos_token + caption + self.tokenizer.eos_token,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            item['input_ids'] = tokenized['input_ids'].squeeze(0)  
            item['attention_mask'] = tokenized['attention_mask'].squeeze(0)
            item['caption_text'] = caption
        
        return item


def get_coco_dataset(
    split: str = 'train',
    transform: Optional[Callable] = None,
    tokenizer: Optional[Callable] = None,
    max_length: int = 77,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    download_images: bool = True
) -> Dataset:
    """
    Factory function to create COCO dataset
    
    Args:
        split (str): Dataset split - 'train', 'val', or 'test'
        transform (Callable, optional): Image transform/preprocessing function
        tokenizer (Callable, optional): Tokenizer for captions
        max_length (int): Maximum caption length
        max_samples (int, optional): Maximum number of samples to load
        cache_dir (str, optional): Directory to cache downloaded images
        download_images (bool): If True, download images from URLs
    
    Returns:
        Dataset: COCO dataset instance
    """
    if tokenizer is not None:
        return COCOKarpathyDatasetWithTokenizer(
            split=split,
            transform=transform,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            cache_dir=cache_dir,
            download_images=download_images
        )
    else:
        return COCOKarpathyDataset(
            split=split,
            transform=transform,
            max_samples=max_samples,
            cache_dir=cache_dir,
            download_images=download_images
        )


if __name__ == "__main__":
    # Example usage
    from torchvision import transforms
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = COCOKarpathyDataset(
        split='train',
        transform=transform,
        max_samples=10  # Load only 10 samples for testing
    )
    
    # Test dataset
    print(f"\nDataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Caption: {sample['caption']}")
    print(f"  Image ID: {sample['image_id']}")
