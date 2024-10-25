import cv2
import numpy as np
from rembg import remove
from PIL import Image
from typing import Tuple, Optional
import argparse
import io

class PhotoProcessor:
    def __init__(self):
        """Initialize the PhotoProcessor with default parameters."""
        # Default parameters for sketch conversion
        self.edge_params = {
            'canny_low': 100,
            'canny_high': 200,
            'sobel_kernel': 3,
            'laplacian_kernel': 3
        }
        
        self.tone_params = {
            'gaussian_kernel': (21, 21),
            'gaussian_sigma': 0,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75
        }
        
        self.texture_params = {
            'block_size': 11,
            'c_value': 2
        }

    def remove_background(
        self,
        image: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Remove background from image and replace with specified color.
        
        Args:
            image: Input image in BGR format
            background_color: RGB color tuple for background
            
        Returns:
            Image with background removed
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Remove background
        output = remove(pil_image)
        
        # Create new image with specified background color
        bg = Image.new('RGBA', output.size, background_color + (255,))
        composite = Image.alpha_composite(bg, output)
        
        # Convert back to OpenCV format (BGR)
        result = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)
        
        return result

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image by applying noise reduction and contrast enhancement.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(
            gray,
            self.tone_params['bilateral_d'],
            self.tone_params['bilateral_sigma_color'],
            self.tone_params['bilateral_sigma_space']
        )
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def create_edge_layer(self, image: np.ndarray) -> np.ndarray:
        """Create edge layer using multiple edge detection methods."""
        # Canny edges
        canny = cv2.Canny(
            image,
            self.edge_params['canny_low'],
            self.edge_params['canny_high']
        )
        
        # Sobel edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.edge_params['sobel_kernel'])
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.edge_params['sobel_kernel'])
        sobel = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        
        # Laplacian edges
        laplacian = cv2.Laplacian(
            image,
            cv2.CV_64F,
            ksize=self.edge_params['laplacian_kernel']
        )
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Combine edge layers
        edges = cv2.addWeighted(canny, 0.4, sobel, 0.3, 0)
        edges = cv2.addWeighted(edges, 0.7, laplacian, 0.3, 0)
        
        return edges

    def create_texture_layer(self, image: np.ndarray) -> np.ndarray:
        """Create texture layer using adaptive thresholding."""
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.texture_params['block_size'],
            self.texture_params['c_value']
        )

    def create_tone_layer(self, image: np.ndarray) -> np.ndarray:
        """Create tone layer using inverted blurring technique."""
        inverted = 255 - image
        blur = cv2.GaussianBlur(
            inverted,
            self.tone_params['gaussian_kernel'],
            self.tone_params['gaussian_sigma']
        )
        return cv2.divide(image, 255 - blur, scale=256.0)

    def create_sketch(
        self,
        image: np.ndarray,
        remove_bg: bool = False,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        style: str = 'detailed',
        include_texture: bool = True,
        include_tones: bool = True
    ) -> np.ndarray:
        """
        Convert photo to sketch with optional background removal.
        
        Args:
            image: Input image in BGR format
            remove_bg: Whether to remove background
            bg_color: Background color (RGB) if removing background
            style: Sketch style ('detailed' or 'simple')
            include_texture: Whether to include texture layer
            include_tones: Whether to include tonal variations
            
        Returns:
            Sketch version of the input image
        """
        # Remove background if requested
        if remove_bg:
            image = self.remove_background(image, bg_color)
        
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Create edge layer
        edges = self.create_edge_layer(processed)
        
        if style == 'simple':
            return edges
        
        # Create and combine additional layers
        layers = [edges]
        
        if include_texture:
            texture = self.create_texture_layer(processed)
            layers.append(texture)
        
        if include_tones:
            tone = self.create_tone_layer(processed)
            layers.append(tone)
        
        # Combine all layers
        sketch = layers[0]
        for layer in layers[1:]:
            sketch = cv2.addWeighted(sketch, 0.7, layer, 0.3, 0)
        
        return sketch

def main():
    """Main function to handle command-line usage."""
    parser = argparse.ArgumentParser(description='Convert photo to sketch with background removal')
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument('output_path', help='Path to save output sketch')
    parser.add_argument(
        '--remove-background',
        action='store_true',
        help='Remove background before conversion'
    )
    parser.add_argument(
        '--background-color',
        type=lambda s: tuple(map(int, s.split(','))),
        default=(255, 255, 255),
        help='Background color as R,G,B (default: 255,255,255)'
    )
    parser.add_argument(
        '--style',
        choices=['detailed', 'simple'],
        default='detailed',
        help='Sketch style to use'
    )
    parser.add_argument(
        '--no-texture',
        action='store_true',
        help='Disable texture layer'
    )
    parser.add_argument(
        '--no-tones',
        action='store_true',
        help='Disable tonal variations'
    )
    
    args = parser.parse_args()
    
    # Validate background color
    if len(args.background_color) != 3:
        print("Error: Background color must be specified as R,G,B")
        return
    
    # Read image
    image = cv2.imread(args.input_path)
    if image is None:
        print(f"Error: Could not read image from {args.input_path}")
        return
    
    # Process image
    processor = PhotoProcessor()
    sketch = processor.create_sketch(
        image,
        remove_bg=args.remove_background,
        bg_color=args.background_color,
        style=args.style,
        include_texture=not args.no_texture,
        include_tones=not args.no_tones
    )
    
    # Save result
    cv2.imwrite(args.output_path, sketch)
    print(f"Sketch saved to {args.output_path}")

if __name__ == "__main__":
    main()