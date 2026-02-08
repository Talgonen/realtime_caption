import tkinter as tk
from PIL import ImageGrab, Image
import threading
import time
from qwen_model import VLModel


class RealtimeCaptionOverlay:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Caption")
        
        # Make window always on top, no decorations, semi-transparent
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.9)
        self.root.overrideredirect(False)  # Keep title bar for dragging
        
        # Position at bottom center of screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 600
        window_height = 80
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 100  # 100px from bottom
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Dark theme
        self.root.configure(bg='#1a1a2e')
        
        # Caption display label
        self.caption_label = tk.Label(
            root,
            text="Loading AI model...",
            font=("Segoe UI", 14, "bold"),
            fg="#ffffff",
            bg='#1a1a2e',
            wraplength=580,
            justify="center"
        )
        self.caption_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Status bar
        self.status_label = tk.Label(
            root,
            text="",
            font=("Segoe UI", 8),
            fg="#888888",
            bg='#1a1a2e'
        )
        self.status_label.pack(side=tk.BOTTOM)
        
        # Control flags
        self.running = True
        self.capturing = False
        self.current_caption = ""
        
        # Load model in background
        threading.Thread(target=self._load_model, daemon=True).start()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Bind keyboard shortcuts
        self.root.bind('<space>', self._toggle_capture)
        self.root.bind('<Escape>', lambda e: self._on_close())
    
    def _load_model(self):
        """Load the AI model in a background thread"""
        try:
            self.model = VLModel()
            self.root.after(0, lambda: self.caption_label.config(text="Press SPACE to start/stop captioning"))
            self.root.after(0, lambda: self.status_label.config(text="Model ready | SPACE: toggle | ESC: quit"))
        except Exception as e:
            self.root.after(0, lambda: self.caption_label.config(text=f"Failed to load model: {str(e)}"))
    
    def _toggle_capture(self, event=None):
        """Toggle the continuous capture on/off"""
        if not hasattr(self, 'model'):
            return
        
        self.capturing = not self.capturing
        if self.capturing:
            self.status_label.config(text="Capturing... | SPACE: pause | ESC: quit", fg="#00ff00")
            threading.Thread(target=self._capture_loop, daemon=True).start()
        else:
            self.status_label.config(text="Paused | SPACE: resume | ESC: quit", fg="#ffaa00")
    
    def _slice_image(self, image, cols=8, rows=4):
        """
        Slice an image into a grid of equally sized rectangles.
        
        Args:
            image: PIL Image to slice
            cols: Number of columns (default 8)
            rows: Number of rows (default 4)
            
        Returns:
            List of (slice_image, position_index) tuples
        """
        width, height = image.size
        slice_width = width // cols
        slice_height = height // rows
        
        slices = []
        for row in range(rows):
            for col in range(cols):
                left = col * slice_width
                upper = row * slice_height
                right = left + slice_width
                lower = upper + slice_height
                
                # Handle edge cases for last row/column
                if col == cols - 1:
                    right = width
                if row == rows - 1:
                    lower = height
                
                slice_img = image.crop((left, upper, right, lower))
                position_idx = row * cols + col
                slices.append((slice_img, position_idx))
        
        return slices
    
    def _capture_loop(self):
        """Continuous capture loop - runs in background thread"""
        while self.running and self.capturing:
            try:
                # Hide window briefly to exclude from capture
                self.root.after(0, self.root.withdraw)
                time.sleep(0.05)  # Brief delay
                
                # Capture screenshot
                screenshot = ImageGrab.grab()
                
                # Restore window
                self.root.after(0, self.root.deiconify)
                
                # Resize for faster processing (don't save)
                max_size = (1024, 768)
                screenshot.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Slice image into 32 rectangles (8x4 grid)
                slices = self._slice_image(screenshot, cols=1, rows=1)
                
                # Extract just the images from slices
                slice_images = [img for img, _ in slices]
                
                # Generate captions for all slices using batch processing
                start_time = time.time()
                captions = self.model.generate_captions_batch(slice_images)
                
                # Combine captions (filter out empty/duplicate ones)
                unique_captions = []
                seen = set()
                for cap in captions:
                    cap_clean = cap.strip().lower()
                    if cap_clean and cap_clean not in seen:
                        seen.add(cap_clean)
                        unique_captions.append(cap.strip())
                
                combined_caption = " | ".join(unique_captions[:5])  # Show top 5 unique
                elapsed = time.time() - start_time
                
                # Update display
                if self.running and self.capturing:
                    self.root.after(0, lambda c=combined_caption: self.caption_label.config(text=c))
                    self.root.after(0, lambda t=elapsed: self.status_label.config(
                        text=f"Updated {t:.1f}s ago (32 slices) | SPACE: pause | ESC: quit"
                    ))
                
                # Wait remaining time to hit ~1 second interval
                # (caption generation already takes some time)
                remaining = max(0, 1.0 - elapsed)
                if remaining > 0 and self.running and self.capturing:
                    time.sleep(remaining)
                    
            except Exception as e:
                self.root.after(0, lambda err=str(e): self.caption_label.config(text=f"Error: {err}"))
                time.sleep(1)
    
    def _on_close(self):
        """Clean shutdown"""
        self.running = False
        self.capturing = False
        self.root.destroy()


def main():
    root = tk.Tk()
    app = RealtimeCaptionOverlay(root)
    root.mainloop()


if __name__ == "__main__":
    main()
