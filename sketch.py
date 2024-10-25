import cv2
import numpy as np
from typing import Tuple, Optional
import argparse

class PhotoToSketch:
    def __init__(self):
        """Initialize the PhotoToSketch converter with default parameters."""
        # Default parameters for various effects
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
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def create_edge_layer(self, image: np.ndarray) -> np.ndarray:
        """
        Create an edge layer using multiple edge detection methods.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Combined edge layer
        """
        # Canny edges
        canny = cv2.Canny(
            image,
            self.edge_params['canny_low'],
            self.edge_params['canny_high']
        )
        
        # Sobel edges (combined X and Y)
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
        """
        Create a texture layer using adaptive thresholding.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Texture layer
        """
        texture = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.texture_params['block_size'],
            self.texture_params['c_value']
        )
        return texture

    def create_tone_layer(self, image: np.ndarray) -> np.ndarray:
        """
        Create a tone layer using inverted blurring technique.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Tone layer
        """
        # Invert image
        inverted = 255 - image
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(
            inverted,
            self.tone_params['gaussian_kernel'],
            self.tone_params['gaussian_sigma']
        )
        
        # Blend inverted and blurred images
        tone = cv2.divide(image, 255 - blur, scale=256.0)
        
        return tone

    def xdog_filter(self, image: np.ndarray, sigma: float = 0.5, k: float = 1.6, p: float = 19) -> np.ndarray:
        """
        Apply Extended Difference of Gaussians (XDoG) filter.
        
        Args:
            image: Preprocessed grayscale image
            sigma: Standard deviation for first Gaussian
            k: Multiplication factor for second Gaussian
            p: Scaling factor for second Gaussian
            
        Returns:
            XDoG filtered image
        """
        # Apply two Gaussian blurs
        g1 = cv2.GaussianBlur(image, (0, 0), sigma)
        g2 = cv2.GaussianBlur(image, (0, 0), sigma * k)
        
        # Calculate XDoG
        dog = g1 - p * g2
        
        # Apply threshold
        edge = 255 * (dog > 0).astype(np.uint8)
        
        return edge

    def create_sketch(
        self,
        image: np.ndarray,
        style: str = 'detailed',
        include_texture: bool = True,
        include_tones: bool = True
    ) -> np.ndarray:
        """
        Convert photo to sketch using specified style and options.
        
        Args:
            image: Input image in BGR format
            style: Sketch style ('detailed', 'simple', or 'xdog')
            include_texture: Whether to include texture layer
            include_tones: Whether to include tonal variations
            
        Returns:
            Sketch version of the input image
        """
        # Preprocess image
        processed = self.preprocess_image(image)
        
        if style == 'xdog':
            # Use XDoG filter for main sketch
            sketch = self.xdog_filter(processed)
        else:
            # Create edge layer
            edges = self.create_edge_layer(processed)
            
            if style == 'simple':
                sketch = edges
            else:  # detailed
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
    parser = argparse.ArgumentParser(description='Convert photo to sketch')
    parser.add_argument('input_path', help='Path to input image')
    parser.add_argument('output_path', help='Path to save output sketch')
    parser.add_argument(
        '--style',
        choices=['detailed', 'simple', 'xdog'],
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
    
    # Read image
    image = cv2.imread(args.input_path)
    if image is None:
        print(f"Error: Could not read image from {args.input_path}")
        return
    
    # Create sketch
    converter = PhotoToSketch()
    sketch = converter.create_sketch(
        image,
        style=args.style,
        include_texture=not args.no_texture,
        include_tones=not args.no_tones
    )
    
    # Save result
    cv2.imwrite(args.output_path, sketch)
    print(f"Sketch saved to {args.output_path}")

if __name__ == "__main__":
    main()