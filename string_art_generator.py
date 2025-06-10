import math
import copy
import random
from collections import Counter # Added for pin usage tracking
from image_processing import calculate_line_darkness, update_image_array

# Originally from stringart.py
def calculate_pins(square_size, radius, num_pins):
    pins = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = square_size // 2 + radius * math.cos(angle)
        y = square_size // 2 + radius * math.sin(angle)
        pins.append((x, y))
    return pins

# Originally from stringart.py
def to_real_coordinates(pin_coords_tuples, square_size, radius_pixel, radius_milimeter):
    """
    Convert pin coordinates from window pixel space to real-world millimeter space.
    Args:
        pin_coords_tuples: List of (x,y) tuples in pixel coordinates.
        square_size: The side length of the square area in pixels.
        radius_pixel: The radius of the pin circle in pixels.
        radius_milimeter: The radius of the pin circle in millimeters.
    Returns:
        List of (x,y) tuples in millimeter coordinates.
    """
    real_pins = []
    center_offset = square_size // 2
    if radius_pixel == 0: # Avoid division by zero
        return [(0,0) for _ in pin_coords_tuples]

    for pin in pin_coords_tuples:
        x = radius_milimeter * (pin[0] - center_offset) / radius_pixel
        y = radius_milimeter * (pin[1] - center_offset) / radius_pixel # Y is often inverted in graphics, but original seems direct
        real_pins.append((x, y))
    return real_pins

# Originally from stringart.py
# Depends on calculate_line_darkness from image_processing.py
def find_best_next_pin(pins_coords, init_array, current_pin_index, last_pin_index, 
                         current_steps_indices, shortest_line_pixels, max_pin_usage):
    """
    Finds the best next pin to draw a line to, based on line darkness and constraints.
    Args:
        pins_coords: List of all pin (x,y) coordinates.
        init_array: Numpy array of the image being processed.
        current_pin_index: Index of the current pin.
        last_pin_index: Index of the previously connected pin (can be None).
        current_steps_indices: List of pin INDICES already used in sequence.
        shortest_line_pixels: Minimum length of a line in pixels.
        max_pin_usage: Maximum number of times a single pin can be part of a line.
    Returns:
        Tuple (best_pin_index, best_line_pixels) or (None, None) if no suitable pin is found.
    """
    best_darkness = 255.0  # Lower is better, ensure float for comparison with new calculate_line_darkness
    best_line_pixels = None
    best_pin_index = None # Return index now
    
    current_pin_coord = pins_coords[current_pin_index]
    pin_usage_counts = Counter(current_steps_indices)

    candidate_indices = []
    for i in range(len(pins_coords)):
        if i == current_pin_index:
            continue
        if last_pin_index is not None and i == last_pin_index: # Avoid going immediately back
            continue
        if pin_usage_counts[i] >= max_pin_usage:
            continue
        candidate_indices.append(i)

    # Consider a random subset for performance if many candidates
    # This sampling logic might need adjustment based on performance after optimization
    if len(candidate_indices) > 30: # Original code sampled 30
        sampled_candidate_indices = random.sample(candidate_indices, 30)
    else:
        sampled_candidate_indices = candidate_indices

    for prospective_pin_index in sampled_candidate_indices:
        prospective_pin_coord = pins_coords[prospective_pin_index]

        # calculate_line_darkness is already optimized
        darkness, line_pixels = calculate_line_darkness(
            init_array,
            int(current_pin_coord[0]), int(current_pin_coord[1]),
            int(prospective_pin_coord[0]), int(prospective_pin_coord[1])
        )

        if darkness < best_darkness and len(line_pixels) >= shortest_line_pixels:
            best_darkness = darkness
            best_line_pixels = line_pixels
            best_pin_index = prospective_pin_index # Store index
            
    return best_pin_index, best_line_pixels, best_darkness

# Originally draw_string from stringart.py, now refactored
# Depends on find_best_next_pin (above) and update_image_array (from image_processing)
def generate_next_string_segment(widgets):
    """
    Calculates the next string segment (line) in the string art generation process.
    This function is responsible for the core logic of choosing the next pin and updating
    the image array representation, but does NOT perform Pygame drawing.

    Args:
        widgets: The main dictionary containing application state, including:
            - "pins": List of (x,y) coordinates for all pins.
            - "current_index": Index of the current pin.
            - "last_index": Index of the last pin connected (can be None).
            - "init_array": Numpy array of the image (modified by update_image_array).
            - "parameters": Dictionary of settings like "Shortest Line in Pixels", "Max Pin Usage".
            - "steps_index": List of pin indices already connected. (Changed from "steps")
            - "processed_array": (Optional) Numpy array of a pre-processed image.
            - "decay": (Optional) Factor for how much lines lighten the image.

    Returns:
        A tuple (next_pin_coord, line_pixel_coords, next_pin_index):
        - next_pin_coord: The (x,y) coordinates of the chosen next pin. (None if no suitable pin)
        - line_pixel_coords: List of (x,y) pixel coordinates for the line to be drawn. (None if no suitable pin)
        - next_pin_index: The index of the chosen next pin. (None if no suitable pin)
    """
    pins = widgets["pins"]
    current_index = widgets["current_index"]
    last_index = widgets["last_index"]
    init_array = widgets["init_array"] 
    parameters = widgets["parameters"]
    # steps = widgets["steps"] # List of pin coordinates - no longer passed directly for usage counting
    steps_indices = widgets["steps_index"] # List of pin indices - THIS IS PASSED NOW

    # Ensure parameters are integers
    shortest_line_pixels_param = parameters.get("Shortest Line in Pixels", 0)
    try:
        shortest_line_pixels = int(shortest_line_pixels_param)
    except ValueError:
        shortest_line_pixels = 0 # Default or log error
    
    max_pin_usage_param = parameters.get("Max Pin Usage", float('inf'))
    try:
        # Check if it's already a number (like float('inf') or int from automated setup)
        if isinstance(max_pin_usage_param, (int, float)):
             max_pin_usage = int(max_pin_usage_param)
        else: # Try to convert from string if it's from UI
             max_pin_usage = int(max_pin_usage_param)
    except ValueError:
        max_pin_usage = float('inf') # Default or log error


    # Call find_best_next_pin with steps_indices
    # It returns: best_pin_index, best_line_pixels, best_darkness (darkness is not used here)
    next_pin_index, line_pixel_coords, _ = find_best_next_pin(
        pins, init_array, current_index, last_index,
        steps_indices, shortest_line_pixels, max_pin_usage # Pass steps_indices
    )

    if next_pin_index is not None and line_pixel_coords: # Check next_pin_index
        update_image_array(widgets, line_pixel_coords)

        # Get next_pin_coord from next_pin_index
        next_pin_coord = pins[next_pin_index] 
                                          
        # The rest of the function (caller responsibilities) remains the same
        # No more pins.index() lookup needed.
        return next_pin_coord, line_pixel_coords, next_pin_index
    else:
        return None, None, None

print("string_art_generator.py populated with functions and imports.")
