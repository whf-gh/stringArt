import math
import copy
import random
from collections import Counter  # Added for pin usage tracking
from image_processing import calculate_line_darkness, update_image_array
import numpy as np

# Simple module-level cache for line computations (Issue 5)
# Key: (min_pin_index, max_pin_index) -> (darkness, line_pixels)
_LINE_CACHE = {}

def reset_line_algorithm_state(widgets=None):
    """Resets cached state for line finding (cache, residuals, angle histogram).
    Optionally clears related entries in provided widgets dict.
    Call when loading a new image, resetting to config, or abandoning a drawing session.
    """
    global _LINE_CACHE
    _LINE_CACHE = {}
    if widgets is not None:
        for key in ["residual_array", "angle_hist"]:
            if key in widgets:
                widgets.pop(key)

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
                       current_steps_indices, minimum_line_spans, max_pin_usage,
                       use_circular_span=False,
                       residual_array=None,
                       angle_hist=None,
                       angle_bins=24,
                       angle_diversity_weight=1.0,
                       exploration_epsilon=0.0,
                       top_k_explore=5):
    """
    Finds the best next pin to draw a line to, based on line darkness and constraints.
    Args:
        pins_coords: List of all pin (x,y) coordinates.
        init_array: Numpy array of the image being processed.
        current_pin_index: Index of the current pin.
        last_pin_index: Index of the previously connected pin (can be None).
        current_steps_indices: Either a list (ordered path of pin indices) OR a Counter of pin usage.
        minimum_line_spans: Minimum pins a line can spans.
        max_pin_usage: Maximum number of times a single pin can be part of a line.
    Returns:
        Tuple (best_pin_index, best_line_pixels) or (None, None) if no suitable pin is found.
    """
    best_darkness = 255.0  # For backward compatibility (if needed externally)
    best_line_pixels = None
    best_pin_index = None  # Return index now

    # Collect scored candidates for epsilon-greedy / diversity adjustments
    scored_candidates = []  # Each element: (score_value, darkness, line_pixels, pin_index)

    num_pins = len(pins_coords)
    current_pin_coord = pins_coords[current_pin_index]

    # Issue 1 fix: accept either list (sequence) or Counter (already counts)
    if isinstance(current_steps_indices, Counter):
        pin_usage_counts = current_steps_indices
        ordered_steps = None  # Can't derive edges reliably
    else:
        # Assume it's an ordered iterable of indices (list/tuple)
        pin_usage_counts = Counter(current_steps_indices)
        ordered_steps = current_steps_indices

    # Issue 6: derive previously used (unordered) edges from ordered steps to avoid reusing identical segments
    used_edges = set()
    if ordered_steps and len(ordered_steps) >= 2:
        for a, b in zip(ordered_steps, ordered_steps[1:]):
            used_edges.add((a, b) if a <= b else (b, a))

    candidate_indices = []
    for i in range(num_pins):
        if i == current_pin_index:
            continue
        if last_pin_index is not None and i == last_pin_index:  # Avoid immediate backtracking
            continue
        if pin_usage_counts[i] >= max_pin_usage:
            continue
        # Skip if edge already used (unordered pair) – Issue 6
        edge_key = (min(current_pin_index, i), max(current_pin_index, i))
        if edge_key in used_edges:
            continue
        candidate_indices.append(i)

    # Performance sampling heuristic remains (could later adapt after caching) – Issue 5 interacts minimally
    if len(candidate_indices) > 30:
        sampled_candidate_indices = random.sample(candidate_indices, 30)
    else:
        sampled_candidate_indices = candidate_indices

    for prospective_pin_index in sampled_candidate_indices:
        prospective_pin_coord = pins_coords[prospective_pin_index]

        # Issue 5: cache line darkness & pixels (unordered edge key)
        cache_key = (id(init_array), min(current_pin_index, prospective_pin_index), max(current_pin_index, prospective_pin_index))
        cached = _LINE_CACHE.get(cache_key)
        if cached is not None:
            darkness, line_pixels = cached
        else:
            darkness, line_pixels = calculate_line_darkness(
                init_array,
                int(current_pin_coord[0]), int(current_pin_coord[1]),
                int(prospective_pin_coord[0]), int(prospective_pin_coord[1])
            )
            if line_pixels:  # cache only meaningful lines
                _LINE_CACHE[cache_key] = (darkness, line_pixels)

        # Issue 2: optionally use circular distance
        index_gap = abs(prospective_pin_index - current_pin_index)
        if use_circular_span and num_pins > 0:
            circular_gap = min(index_gap, num_pins - index_gap)
            gap_metric = circular_gap
        else:
            gap_metric = index_gap

        if gap_metric < minimum_line_spans:
            continue

        # Compute base contribution score
        if not line_pixels:  # Skip candidates with no valid pixels (keeps legacy behavior of ignoring empty lines)
            continue
        if residual_array is not None and line_pixels:
            # Sum residual deficits: residual assumed = 255 - init_array
            ys = [p[1] for p in line_pixels]
            xs = [p[0] for p in line_pixels]
            try:
                deficits = residual_array[ys, xs]
                base_score = float(np.sum(deficits))  # Larger is better
            except Exception:
                base_score = float(len(line_pixels))  # Fallback minimal contribution
        else:
            # Fallback: use inverse of darkness * length
            length_factor = len(line_pixels) if line_pixels else 1
            base_score = (255.0 - darkness) * length_factor

        # Angle diversity penalty (histogram of recent angles)
        if line_pixels:
            dx = pins_coords[prospective_pin_index][0] - pins_coords[current_pin_index][0]
            dy = pins_coords[prospective_pin_index][1] - pins_coords[current_pin_index][1]
            angle = math.atan2(dy, dx)  # range [-pi, pi]
            bin_index = int(((angle + math.pi) / (2 * math.pi)) * angle_bins) % angle_bins
            if angle_hist is not None and angle_bins > 0:
                usage = angle_hist.get(bin_index, 0)
                diversity_multiplier = 1.0 / (1.0 + angle_diversity_weight * usage)
            else:
                diversity_multiplier = 1.0
        else:
            diversity_multiplier = 1.0

        final_score = base_score * diversity_multiplier
        scored_candidates.append((final_score, darkness, line_pixels, prospective_pin_index))

    if not scored_candidates:
        return None, None, best_darkness

    # Sort by descending score (higher is better now)
    scored_candidates.sort(key=lambda t: t[0], reverse=True)

    # Epsilon-greedy exploration among top_k_explore
    explore_pool = scored_candidates[:max(1, top_k_explore)]
    if exploration_epsilon > 0.0 and random.random() < exploration_epsilon and len(explore_pool) > 1:
        chosen = random.choice(explore_pool[1:])  # Avoid always picking best when exploring
    else:
        chosen = explore_pool[0]

    final_score, chosen_darkness, chosen_pixels, chosen_pin = chosen
    best_darkness = chosen_darkness
    best_line_pixels = chosen_pixels
    best_pin_index = chosen_pin

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


    # Call find_best_next_pin with steps_indices.
    # It returns: best_pin_index, best_line_pixels, best_darkness (darkness is not used here)
    use_circular_span = bool(parameters.get("Use Circular Span", False))

    # Residual array: maintain or create (255 - init_array) once per call
    residual_array = widgets.get("residual_array")
    if residual_array is None or residual_array.shape != init_array.shape:
        residual_array = 255 - init_array
        widgets["residual_array"] = residual_array

    # Angle histogram tracking
    angle_hist = widgets.get("angle_hist")
    if angle_hist is None:
        angle_hist = {}
        widgets["angle_hist"] = angle_hist
    angle_bins = int(parameters.get("Angle Bins", 24) or 24)
    angle_diversity_weight = float(parameters.get("Angle Diversity Weight", 1.0) or 1.0)

    # Exploration parameters
    exploration_epsilon = float(parameters.get("Exploration Epsilon", 0.05) or 0.0)
    top_k_explore = int(parameters.get("Top K Explore", 5) or 5)

    next_pin_index, line_pixel_coords, _ = find_best_next_pin(
        pins, init_array, current_index, last_index,
        steps_indices, minimum_line_spans, max_pin_usage,
        use_circular_span=use_circular_span,
        residual_array=residual_array,
        angle_hist=angle_hist,
        angle_bins=angle_bins,
        angle_diversity_weight=angle_diversity_weight,
        exploration_epsilon=exploration_epsilon,
        top_k_explore=top_k_explore
    )

    if next_pin_index is not None and line_pixel_coords: # Check next_pin_index
        update_image_array(widgets, line_pixel_coords)

        # Update residual array after progressive lightening
        updated_array = widgets["init_array"]
        # Recompute only affected pixels for efficiency
        if "residual_array" in widgets:
            for x, y in line_pixel_coords:
                if 0 <= y < updated_array.shape[0] and 0 <= x < updated_array.shape[1]:
                    widgets["residual_array"][y, x] = 255 - updated_array[y, x]

        # Update angle histogram
        if line_pixel_coords:
            # Derive angle from endpoints
            pin_a = pins[current_index]
            pin_b = pins[next_pin_index]
            dx = pin_b[0] - pin_a[0]
            dy = pin_b[1] - pin_a[1]
            angle = math.atan2(dy, dx)
            bin_index = int(((angle + math.pi) / (2 * math.pi)) * angle_bins) % angle_bins
            angle_hist[bin_index] = angle_hist.get(bin_index, 0) + 1

        # Maintain ordered steps list (needed for edge reuse avoidance)
        # Caller previously handled this; we defensively update if list present.
        if isinstance(steps_indices, list):
            steps_indices.append(next_pin_index)
            widgets["steps_index"] = steps_indices

        # Get next_pin_coord from next_pin_index
        next_pin_coord = pins[next_pin_index]

        return next_pin_coord, line_pixel_coords, next_pin_index
    else:
        return None, None, None

print("string_art_generator.py populated with functions and imports.")
