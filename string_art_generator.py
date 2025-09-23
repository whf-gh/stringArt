import math
import copy
import random
from collections import Counter # Added for pin usage tracking
from image_processing import calculate_line_darkness, update_image_array
import numpy as np

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


def calculate_angular_distance(current_index, prospective_index, total_pins):
    """
    Calculate the angular distance between two pins on a circle.
    Returns the distance in pin steps (minimum of clockwise and counter-clockwise).
    """
    clockwise = (prospective_index - current_index) % total_pins
    counter_clockwise = (current_index - prospective_index) % total_pins
    return min(clockwise, counter_clockwise)


def calculate_distance_based_sampling(pins_coords, current_pin_index, candidate_indices,
                                     total_pins, sample_size=30, min_distance_factor=0.1,
                                     max_distance_factor=0.9):
    """
    Select candidate pins using distance-based sampling instead of random sampling.
    Prioritizes pins that are neither too close nor too far from the current pin.
    """
    if len(candidate_indices) <= sample_size:
        return candidate_indices

    # Calculate angular distance for each candidate
    distances = []
    for idx in candidate_indices:
        angular_distance = calculate_angular_distance(current_pin_index, idx, total_pins)
        distances.append((idx, angular_distance))

    # Convert angular distance to actual pixel distance for better sampling
    pin_coords = pins_coords[current_pin_index]
    distances_with_pixel = []
    for idx, angular_dist in distances:
        prospective_coords = pins_coords[idx]
        # Calculate Euclidean distance between pins
        pixel_distance = math.sqrt((pin_coords[0] - prospective_coords[0])**2 +
                                 (pin_coords[1] - prospective_coords[1])**2)
        distances_with_pixel.append((idx, angular_dist, pixel_distance))

    # Define optimal distance range (avoid very short and very long lines)
    max_possible_distance = min(pins_coords[0][0] * 2, pins_coords[0][1] * 2)  # rough diameter
    min_distance = max_possible_distance * min_distance_factor
    max_distance = max_possible_distance * max_distance_factor

    # Score candidates based on distance (higher score for distances in optimal range)
    scored_candidates = []
    for idx, angular_dist, pixel_distance in distances_with_pixel:
        # Score based on pixel distance (prefer distances in optimal range)
        if pixel_distance < min_distance:
            distance_score = 0.2  # Low score for very short lines
        elif pixel_distance > max_distance:
            distance_score = 0.3  # Medium-low score for very long lines
        elif max_distance * 0.7 <= pixel_distance <= max_distance:
            distance_score = 0.8  # High score for long lines in optimal range
        elif min_distance <= pixel_distance <= max_distance * 0.7:
            distance_score = 1.0  # Highest score for medium lines
        else:
            distance_score = 0.5  # Medium score for others

        # Bonus for moderate angular distances (avoid adjacent pins)
        angular_score = 1.0 - abs(0.5 - (angular_dist / (total_pins // 2)))

        # Combine scores
        final_score = distance_score * 0.7 + angular_score * 0.3
        scored_candidates.append((idx, final_score))

    # Sort by score (descending) and take top samples
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Take top candidates, but ensure some diversity by also including some lower-scored ones
    top_candidates = [idx for idx, score in scored_candidates[:int(sample_size * 0.8)]]

    # Add some diversity by including some lower-scored candidates
    remaining_candidates = [idx for idx, score in scored_candidates[int(sample_size * 0.8):]]
    if len(remaining_candidates) > int(sample_size * 0.2):
        diverse_candidates = random.sample(remaining_candidates, int(sample_size * 0.2))
    else:
        diverse_candidates = remaining_candidates

    return top_candidates + diverse_candidates

# Originally from stringart.py
# Depends on calculate_line_darkness from image_processing.py
def find_best_next_pin(pins_coords, init_array, current_pin_index, last_pin_index,
                         current_steps_indices, minimum_line_spans, max_pin_usage,
                         temperature=1.0):
    """
    Finds the best next pin to draw a line to, based on line darkness and constraints.
    Args:
        pins_coords: List of all pin (x,y) coordinates.
        init_array: Numpy array of the image being processed.
        current_pin_index: Index of the current pin.
        last_pin_index: Index of the previously connected pin (can be None).
        current_steps_indices: List of pin INDICES already used in sequence.
        minimum_line_spans: Minimum pins a line can spans.
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

    # Use distance-based sampling instead of random sampling
    # This prioritizes pins at optimal distances for better line quality
    if len(candidate_indices) > 10:  # Only sample if we have enough candidates
        sampled_candidate_indices = calculate_distance_based_sampling(
            pins_coords, current_pin_index, candidate_indices, len(pins_coords), sample_size=30
        )
    else:
        sampled_candidate_indices = candidate_indices

    for prospective_pin_index in sampled_candidate_indices:
        prospective_pin_coord = pins_coords[prospective_pin_index]

        # calculate_line_darkness is already optimized
        base_darkness, line_pixels = calculate_line_darkness(
            init_array,
            int(current_pin_coord[0]), int(current_pin_coord[1]),
            int(prospective_pin_coord[0]), int(prospective_pin_coord[1])
        )

        # Apply weighted darkness calculation for better line quality
        # This gives higher weight to pixels in the center of the line
        if line_pixels and len(line_pixels) > 2:
            # Get pixel values along the line
            rr, cc = zip(*[(int(y), int(x)) for x, y in line_pixels])
            pixel_values = init_array[rr, cc]

            # Apply weights: higher weight for center pixels, lower for endpoints
            line_length = len(pixel_values)
            weights = []

            for i in range(line_length):
                # Weight based on distance from center (0 = endpoint, 1 = center)
                distance_from_center = abs(i - line_length / 2) / (line_length / 2)
                # Higher weight for center, lower for endpoints (quadratic curve)
                center_weight = max(0.3, 1.0 - distance_from_center * distance_from_center)
                weights.append(center_weight)

            # Normalize weights
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]

            # Calculate weighted darkness
            darkness = sum(p * w for p, w in zip(pixel_values, normalized_weights))
        else:
            # Fall back to base darkness for very short lines
            darkness = base_darkness

        # Apply simulated annealing if temperature is provided (for global optimization)
        if temperature > 1.0:
            # Occasionally accept worse solutions to escape local optima
            acceptance_probability = math.exp((best_darkness - darkness) / temperature)
            if random.random() < acceptance_probability:
                best_darkness = darkness
                best_line_pixels = line_pixels
                best_pin_index = prospective_pin_index # Store index
 # Always accept better solutions
        elif darkness < best_darkness:
            best_darkness = darkness
            best_line_pixels = line_pixels
            best_pin_index = prospective_pin_index  # Store index
        else:
            # Standard greedy selection
            if darkness < best_darkness and abs(prospective_pin_index - current_pin_index) >= minimum_line_spans:
                best_darkness = darkness
                best_line_pixels = line_pixels
                best_pin_index = prospective_pin_index  # Store index
            
    return best_pin_index, best_line_pixels, best_darkness

def find_best_next_pin_2_lookahead(pins_coords, init_array, current_pin_index, last_pin_index,
 current_steps_indices, minimum_line_spans, max_pin_usage,
                                    temperature=1.0, lookahead_weight=0.3):
    """
    Enhanced version of find_best_next_pin with 2-pin lookahead.
    This function considers not just the immediate next pin, but also the quality
    of the potential line after that, leading to better overall sequences.

    Args:
        pins_coords: List of all pin (x,y) coordinates.
        init_array: Numpy array of the image being processed.
        current_pin_index: Index of the current pin.
        last_pin_index: Index of the previously connected pin (can be None).
        current_steps_indices: List of pin INDICES already used in sequence.
        minimum_line_spans: Minimum pins a line can spans.
        max_pin_usage: Maximum number of times a single pin can be part of a line.
        temperature: Temperature for simulated annealing (1.0 = greedy).
        lookahead_weight: Weight given to the lookahead evaluation (0-1).

    Returns:
        Tuple (best_pin_index, best_line_pixels, best_darkness)
    """
    # Get immediate best options
    best_pin_index, best_line_pixels, best_darkness = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        current_steps_indices, minimum_line_spans, max_pin_usage, temperature
    )

    if best_pin_index is None or len(current_steps_indices) == 0:
 # No lookahead possible or no previous steps
        return best_pin_index, best_line_pixels, best_darkness

    # Create a temporary copy of the image array to simulate the first line
    temp_array = init_array.copy()

    # Simulate drawing the first line to update the temporary image
    for x, y in best_line_pixels:
        if 0 <= y < temp_array.shape[0] and 0 <= x < temp_array.shape[1]:
            if temp_array[y, x] < 200:
                temp_array[y, x] = min(int(temp_array[y, x] + 100), 255)
            else:
                temp_array[y, x] = 255

    # Find the best second line from the prospective next pin
    pin_usage_counts = current_steps_indices + [best_pin_index]

    # Get candidates for the second line (excluding the current and prospective pins)
    second_candidates = []
    for i in range(len(pins_coords)):
        if i == best_pin_index:  # Can't go to the same pin
            continue
        if i == current_pin_index:  # Can't go back to current immediately
            continue
        if i in pin_usage_counts and pin_usage_counts.count(i) >= max_pin_usage:
            continue
        if abs(i - best_pin_index) < minimum_line_spans:
            continue
        second_candidates.append(i)

    # If we have candidates, evaluate the lookahead quality
    if len(second_candidates) > 0:
        # Use distance-based sampling for lookahead candidates
        sampled_second_candidates = calculate_distance_based_sampling(
            pins_coords, best_pin_index, second_candidates, len(pins_coords), sample_size=15
        )

        best_lookahead_darkness = 255.0

        # Evaluate each second candidate
        for second_pin_index in sampled_second_candidates:
            second_pin_coord = pins_coords[second_pin_index]

            lookahead_darkness, _ = calculate_line_darkness(
                temp_array,
                int(pins_coords[best_pin_index][0]), int(pins_coords[best_pin_index][1]),
                int(second_pin_coord[0]), int(second_pin_coord[1])
            )

            if lookahead_darkness < best_lookahead_darkness:
                best_lookahead_darkness = lookahead_darkness

        # Combine immediate and lookahead evaluations
        # Lower combined score is better
        combined_score = best_darkness * (1 - lookahead_weight) + best_lookahead_darkness * lookahead_weight

        # Re-evaluate all immediate candidates with lookahead
        pin_usage_counts_base = Counter(current_steps_indices)
        immediate_candidates = []
        for i in range(len(pins_coords)):
            if i == current_pin_index:
                continue
            if last_pin_index is not None and i == last_pin_index:
                continue
            if pin_usage_counts_base[i] >= max_pin_usage:
                continue
            if abs(i - current_pin_index) < minimum_line_spans:
                continue
            immediate_candidates.append(i)

        # Use distance-based sampling
        sampled_immediate = calculate_distance_based_sampling(
            pins_coords, current_pin_index, immediate_candidates, len(pins_coords), sample_size=20
        )

        best_combined_score = combined_score

        # Define current_pin_coord for use in this scope
        current_pin_coord = pins_coords[current_pin_index]

        for immediate_pin_index in sampled_immediate:
            if immediate_pin_index == best_pin_index:
                continue # Skip the one we already evaluated

            immediate_pin_coord = pins_coords[immediate_pin_index]

            # Calculate immediate darkness
            immediate_darkness, immediate_line_pixels = calculate_line_darkness(
                init_array,
                int(current_pin_coord[0]), int(current_pin_coord[1]),
                int(immediate_pin_coord[0]), int(immediate_pin_coord[1])
            )

            # Simulate first line
            temp_array_immediate = init_array.copy()
            for x, y in immediate_line_pixels:
                if 0 <= y < temp_array_immediate.shape[0] and 0 <= x < temp_array_immediate.shape[1]:
                    if temp_array_immediate[y, x] < 200:
                        temp_array_immediate[y, x] = min(int(temp_array_immediate[y, x] + 100), 255)
                    else:
                        temp_array_immediate[y, x] = 255

            # Find best second line from this pin
            second_candidates_immediate = []
            pin_usage_counts_immediate = current_steps_indices + [immediate_pin_index]

            for i in range(len(pins_coords)):
                if i == immediate_pin_index:
                    continue
                if i == current_pin_index:
                    continue
                if i in pin_usage_counts_immediate and pin_usage_counts_immediate.count(i) >= max_pin_usage:
                    continue
                if abs(i - immediate_pin_index) < minimum_line_spans:
                    continue
                second_candidates_immediate.append(i)

            if len(second_candidates_immediate) > 0:
                # Sample and find best second line
                sampled_second_immediate = calculate_distance_based_sampling(
                    pins_coords, immediate_pin_index, second_candidates_immediate,
                    len(pins_coords), sample_size=10
                )

                best_lookahead_for_immediate = 255.0
                for second_idx in sampled_second_immediate:
                    second_coord = pins_coords[second_idx]
                    lookahead_dark, _ = calculate_line_darkness(
                        temp_array_immediate,
                        int(immediate_pin_coord[0]), int(immediate_pin_coord[1]),
                        int(second_coord[0]), int(second_coord[1])
                    )

                    if lookahead_dark < best_lookahead_for_immediate:
                        best_lookahead_for_immediate = lookahead_dark

                # Combined score for this immediate choice
                immediate_combined = immediate_darkness * (1 - lookahead_weight) + best_lookahead_for_immediate * lookahead_weight

                # If this combination is better, use it instead
                if immediate_combined < best_combined_score:
                    best_combined_score = immediate_combined
                    best_darkness = immediate_darkness
                    best_line_pixels = immediate_line_pixels
                    best_pin_index = immediate_pin_index

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
            - "parameters": Dictionary of settings like "Minimum Pins Line Spans", "Max Pin Usage".
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
    minimum_line_spans_param = parameters.get("Minimum Pins Line Spans", 0)
    try:
        minimum_line_spans = int(minimum_line_spans_param)
    except ValueError:
        minimum_line_spans = 0 # Default or log error
    
    max_pin_usage_param = parameters.get("Max Pin Usage", float('inf'))
    try:
        # Check if it's already a number (like float('inf') or int from automated setup)
        if isinstance(max_pin_usage_param, (int, float)):
             max_pin_usage = int(max_pin_usage_param)
        else: # Try to convert from string if it's from UI
             max_pin_usage = int(max_pin_usage_param)
    except ValueError:
        max_pin_usage = float('inf') # Default or log error


    # Add temperature parameter for simulated annealing
    # Temperature decreases as we progress through the drawing (starts at 2.0, decreases to 1.0)
    total_steps = parameters.get("Number of Lines", 100)
    current_step_count = len(steps_indices)
    temperature = max(1.0, 2.0 - (current_step_count / total_steps) * 1.0) if total_steps > 0 else 1.0

    # Determine which algorithm to use based on progress
    # Use lookahead for early stages, switch to faster greedy for later stages
    use_lookahead = current_step_count < total_steps * 0.7  # Use lookahead for first 70%

    # Call the appropriate pin finding algorithm
    # It returns: best_pin_index, best_line_pixels, best_darkness (darkness is not used here)
    if use_lookahead and len(pins) > 10:  # Only use lookahead if we have enough pins
        next_pin_index, line_pixel_coords, _ = find_best_next_pin_2_lookahead(
            pins, init_array, current_index, last_index,
            steps_indices, minimum_line_spans, max_pin_usage,
            temperature=temperature, lookahead_weight=0.3
        )
    else:
        # Use the enhanced greedy algorithm for later stages (faster)
        next_pin_index, line_pixel_coords, _ = find_best_next_pin(
            pins, init_array, current_index, last_index,
            steps_indices, minimum_line_spans, max_pin_usage,
            temperature=temperature
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
