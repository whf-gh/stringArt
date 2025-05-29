import cv2
import numpy as np
from PIL import Image, ImageDraw
from tkinter import Tk, filedialog
from rembg import remove
from skimage import draw as skidraw # Added for optimized line calculation

# Originally from stringart.py
def calculate_line_darkness(img_array, x1, y1, x2, y2):
    """
    Calculate the darkness of pixel values along a line between two points in a grayscale image.

    Args:
        img_array: numpy array of the image (grayscale)
        x1, y1: Starting point coordinates
        x2, y2: Ending point coordinates

    Returns:
        tuple: (darkness of pixel values, list of coordinates along the line)
    """
    pixels = []
    pixel_sum = 0

    # skimage.draw.line uses (row, col) which corresponds to (y, x)
    rr, cc = skidraw.line(int(y1), int(x1), int(y2), int(x2))

    # Ensure coordinates are within image bounds
    height, width = img_array.shape
    valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
    
    rr_valid = rr[valid_indices]
    cc_valid = cc[valid_indices]

    if rr_valid.size == 0: # No pixels of the line are within the image
        return 255.0, []

    # Extract pixel values along the line
    # img_array values are typically uint8. Summing them can exceed 255.
    # np.sum will use a larger dtype for accumulation by default (e.g., np.int64 for uint8).
    pixel_sum = np.sum(img_array[rr_valid, cc_valid])
    
    num_pixels = rr_valid.size
    average_darkness = pixel_sum / num_pixels if num_pixels > 0 else 255.0

    # Create list of (x,y) coordinates (ensure they are Python ints for consistency if needed by other parts)
    # The original function returned a list of tuples: pixels.append((x, y))
    line_pixels = list(zip(cc_valid.tolist(), rr_valid.tolist()))

    return average_darkness, line_pixels


# Originally from stringart.py
def update_image_array(widgets, line): # widgets here is the main dictionary
    # This function modifies widgets['init_array'] based on the line drawn.
    # It's part of the core string art algorithm's image update logic.
    init_array = widgets["init_array"]
    processed_array = widgets.get("processed_array") # Might be None

    for x, y in line:
        if 0 <= y < init_array.shape[0] and 0 <= x < init_array.shape[1]: # Boundary check
            if init_array[y, x] != 255: # If not already white
                if processed_array is not None:
                    if processed_array[y, x] == 0 and init_array[y, x] < 200:
                        init_array[y, x] = (
                            init_array[y, x]
                            + (255 - init_array[y, x]) // 2
                        )
                    else:
                        init_array[y, x] = 255
                elif init_array[y, x] < 200:
                    # Original logic: widgets['init_array'][y, x] = min(int(widgets['init_array'][y, x] * widgets['decay']), 255)
                    # Using a simpler update for now if decay is not passed:
                    init_array[y, x] = min(int(init_array[y, x] + 100), 255)
                else:
                    init_array[y, x] = 255
    widgets["init_array"] = init_array # Ensure the main widgets dict is updated


# Originally from ui.py, now in image_processing.py
def select_image(image_square_size, current_image_path=None):
    """
    Opens a file dialog for image selection, resizes and crops the image.
    Does not directly modify 'widgets' dict but returns image objects.
    """
    image = None
    image_original = None
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialfile=current_image_path)
    if file_path:
        pil_image = Image.open(file_path)
        img_width, img_height = pil_image.size

        if img_width < image_square_size or img_height < image_square_size:
            scale_factor = max(image_square_size / img_width, image_square_size / img_height)
            new_size = (int(img_width * scale_factor), int(img_height * scale_factor))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
            img_width, img_height = new_size

        if img_width > image_square_size or img_height > image_square_size:
            left = (img_width - image_square_size) // 2
            top = (img_height - image_square_size) // 2
            right = left + image_square_size
            bottom = top + image_square_size
            pil_image = pil_image.crop((left, top, right, bottom))
        
        image = pil_image # This is the potentially processed (cropped/resized) image
        image_original = Image.open(file_path).copy() # Store a copy of the original for resets
                                                 # Or, if the above pil_image is what we want as original after first load:
                                                 # image_original = pil_image.copy()
                                                 # For consistency with original code, let's assume 'image' becomes the new 'image'
                                                 # and 'image_original' is for reverting processing steps, not selection.
                                                 # The original code did: widgets["image"] = image; widgets["image_original"] = image.copy()
                                                 # after crop/resize. This implies image_original is the cropped/resized one.
        image_original = image.copy()


    root.destroy()
    # Returns the selected and processed (resized/cropped) PIL image, and its copy.
    # The main application (stringart.py) will be responsible for putting these into the 'widgets' dictionary.
    return image, image_original


# Originally from ui.py, now in image_processing.py
def apply_circle_mask(image, center, radius):
    """Applies a circular mask to a PIL image."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)],
        fill=255,
    )
    image.putalpha(mask)
    # Create a new RGB image with a white background
    white_background = Image.new("RGB", image.size, (255, 255, 255))
    # Paste the image with alpha mask onto the white background
    white_background.paste(image, (0,0), image) # Use the image's alpha channel as the mask
    return white_background # Return RGB image

# Originally from ui.py, now in image_processing.py
def process_image(image_to_process, checkboxes_state, image_square_size, radius_pixels, existing_init_array=None):
    """
    Processes a PIL image based on checkbox states.
    Returns new 'image' (PIL, processed), 'init_array' (numpy), 'processed_array' (numpy).
    'image_to_process' is equivalent to widgets["image_original"] (the clean, selected image).
    'checkboxes_state' is equivalent to widgets["checkboxes"].
    """
    edge_params = {
        "canny_low": 100, "canny_high": 160, "adaptive_block": 9, "adaptive_c": 2,
        "blur_kernel": (5, 5), "dilate_kernel": np.ones((2, 2), np.uint8),
        "erode_kernel": np.ones((1, 1), np.uint8),
    }

    # Start with a copy of the original image for processing
    img = image_to_process.copy()

    if checkboxes_state.get("Remove Background", False):
        img = remove(img)
        # Ensure background is white after removal (if alpha channel is present)
        if img.mode == 'RGBA':
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        else: # If no alpha, ensure it's RGB for consistency before L conversion
            img = img.convert('RGB')


    img_l = img.convert("L") # Convert to grayscale
    image_array = np.array(img_l)
    
    # This will be the basis for init_array if not using processed image only later
    # Or if an existing_init_array (e.g. from a previous load before reprocessing) is not supplied.
    # The original code did: widgets["init_array"] = image_array.copy() early on.
    # We'll return it, stringart.py can decide.
    current_init_array_candidate = image_array.copy()


    if checkboxes_state.get("denoise", False):
        image_array = cv2.bilateralFilter(image_array, 9, 75, 75)
    if checkboxes_state.get("Canny edge detection", False):
        image_array = cv2.Canny(image_array, edge_params["canny_low"], edge_params["canny_high"])
    if checkboxes_state.get("Adaptive thresholding", False):
        image_array = cv2.adaptiveThreshold(
            image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, edge_params["adaptive_block"], edge_params["adaptive_c"]
        )
    if checkboxes_state.get("Global Thresholding", False):
        _, image_array = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
    if checkboxes_state.get("Line Thickness", False):
        image_array = cv2.dilate(image_array, edge_params["dilate_kernel"], iterations=1) # original had 3-1=2

    if image_array.size > 0 and np.any(image_array): # Check if not empty and not all white
      if image_array.ndim == 2 and image_array[0,0] == 0: # Check for inversion for B/W images
          # This condition might need to be more robust, e.g., check average color
          # For now, keeping it similar to original if first pixel is black
          image_array = cv2.bitwise_not(image_array)
    
    # Convert the processed numpy array back to a PIL Image
    processed_pil_image = Image.fromarray(image_array)

    # Apply circle mask to the processed PIL image
    final_pil_image = apply_circle_mask(
        processed_pil_image,
        (image_square_size // 2, image_square_size // 2),
        radius_pixels
    )

    # This is the array from the final visually processed image (after masking)
    final_processed_array = np.array(final_pil_image.convert("L"))

    # Determine the init_array based on "Use Processed Image Only"
    # The original init_array (before any processing, but after selection & resize/crop)
    # also needs masking for consistent comparison/display.
    # If existing_init_array is passed (e.g. widgets["image_original"] converted to array), use that as base.
    # Otherwise, use the current_init_array_candidate.
    base_for_init_array_pil = Image.fromarray(existing_init_array if existing_init_array is not None else current_init_array_candidate)
    
    masked_original_init_array = np.array(
        apply_circle_mask(
            base_for_init_array_pil,
            (image_square_size // 2, image_square_size // 2),
            radius_pixels
        ).convert("L")
    )

    init_array_to_use = final_processed_array.copy() if checkboxes_state.get("Use Processed Image Only", False) else masked_original_init_array
    
    # If "Use Processed Image Only" is true, processed_array for drawing logic might be considered None by stringart.py
    # otherwise, it's the result of edge detection etc.
    output_processed_array = None if checkboxes_state.get("Use Processed Image Only", False) else final_processed_array

    # final_pil_image is the image to be displayed in preview (widgets["image"])
    # init_array_to_use is what string drawing logic (find_best_next_pin) will use (widgets["init_array"])
    # output_processed_array is for secondary drawing logic or display (widgets["processed_array"])
    return final_pil_image, init_array_to_use, output_processed_array

print("image_processing.py populated with functions and imports.")
