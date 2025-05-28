import math
import copy
import random
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
                         current_steps_pins, shortest_line_pixels, max_pin_usage):
    """
    Finds the best next pin to draw a line to, based on line darkness and constraints.
    Args:
        pins_coords: List of all pin (x,y) coordinates.
        init_array: Numpy array of the image being processed.
        current_pin_index: Index of the current pin.
        last_pin_index: Index of the previously connected pin (can be None).
        current_steps_pins: List of pin (x,y) coordinates already used in sequence.
        shortest_line_pixels: Minimum length of a line in pixels.
        max_pin_usage: Maximum number of times a single pin can be part of a line.
    Returns:
        Tuple (best_pin_coord, best_line_pixels) or (None, None) if no suitable pin is found.
    """
    best_darkness = 255  # Lower is better
    best_line_pixels = None
    best_pin_coord = None
    
    current_pin_coord = pins_coords[current_pin_index]
    
    # Create a mutable copy of all pin coordinates to select from
    target_pin_coords = list(pins_coords) 

    # Exclude the current pin itself
    # Popping by index needs care if list is modified. Better to build a list of candidates.
    candidate_pins = []
    for i, pin_coord in enumerate(target_pin_coords):
        if i == current_pin_index:
            continue
        if last_pin_index is not None and i == last_pin_index: # Avoid going immediately back
            continue
        candidate_pins.append(pin_coord)

    # Further filtering based on usage (complex conditions from original):
    # The original logic for 'steps' and removing pins already connected to current_pin in 'steps'
    # is tricky. Let's simplify: we primarily check against max_pin_usage for any candidate.
    # A more direct interpretation of "remove used steps" from original:
    # "target_pins" were reduced by removing pins adjacent to ANY instance of current_pin in steps.
    # This seems too restrictive. Let's focus on max_pin_usage and not reusing immediate neighbors if possible.
    
    # Consider a random subset for performance, as in original
    # If many candidates, sample randomly. Otherwise, use all.
    if len(candidate_pins) > 30:
        sampled_candidate_pins = random.sample(candidate_pins, 30)
    else:
        sampled_candidate_pins = candidate_pins

    for prospective_pin_coord in sampled_candidate_pins:
        # Check max_pin_usage: Count how many times prospective_pin_coord appears in current_steps_pins
        # Note: current_steps_pins contains coordinates, not indices.
        if current_steps_pins.count(prospective_pin_coord) >= max_pin_usage:
            continue

        darkness, line_pixels = calculate_line_darkness(
            init_array,
            int(current_pin_coord[0]), int(current_pin_coord[1]),
            int(prospective_pin_coord[0]), int(prospective_pin_coord[1])
        )

        if darkness < best_darkness and len(line_pixels) >= shortest_line_pixels:
            best_darkness = darkness
            best_line_pixels = line_pixels
            best_pin_coord = prospective_pin_coord
            
    return best_pin_coord, best_line_pixels

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
            - "steps": List of (x,y) pin coordinates already connected.
            - "steps_index": List of pin indices already connected.
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
    init_array = widgets["init_array"] # This will be modified by update_image_array
    parameters = widgets["parameters"]
    steps = widgets["steps"] # List of pin coordinates
    # steps_index = widgets["steps_index"] # List of pin indices

    shortest_line_pixels = parameters.get("Shortest Line in Pixels", 0)
    max_pin_usage = parameters.get("Max Pin Usage", float('inf')) # Default to no limit if not specified

    # Call find_best_next_pin
    # find_best_next_pin expects: pins_coords, init_array, current_pin_index, last_pin_index, 
    #                            current_steps_pins, shortest_line_pixels, max_pin_usage
    next_pin_coord, line_pixel_coords = find_best_next_pin(
        pins, init_array, current_index, last_index,
        steps, shortest_line_pixels, max_pin_usage
    )

    if next_pin_coord is not None and line_pixel_coords:
        # If a next pin is found, update the image array
        # update_image_array modifies widgets["init_array"] directly or returns the modified array.
        # Current signature is update_image_array(widgets, line), it modifies widgets['init_array']
        update_image_array(widgets, line_pixel_coords) # Pass the whole widgets dict as it expects it

        # Determine the index of the next_pin_coord
        try:
            next_pin_index = pins.index(next_pin_coord)
        except ValueError:
            # This should not happen if next_pin_coord comes from the pins list
            return None, None, None 

        # The caller (stringart.py) will be responsible for:
        # - Updating widgets["last_index"] = current_index
        # - Updating widgets["current_index"] = next_pin_index
        # - Appending next_pin_coord to widgets["steps"]
        # - Appending next_pin_index to widgets["steps_index"]
        # - Updating widgets["total_length_in_pixels"] += len(line_pixel_coords)
        # - Handling the actual drawing on the Pygame surface.
        
        return next_pin_coord, line_pixel_coords, next_pin_index
    else:
        return None, None, None

print("string_art_generator.py populated with functions and imports.")
