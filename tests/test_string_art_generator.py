import math
import pytest
import numpy as np # Not strictly needed for these initial tests, but good for consistency
import sys
import os

# Add the project root to the Python path to allow importing from string_art_generator
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from string_art_generator import calculate_pins, to_real_coordinates, find_best_next_pin, generate_next_string_segment
# For mocking and setup, we might need these from image_processing
# Note: calculate_line_darkness and update_image_array are used by string_art_generator internally.
# We might need to mock them directly where they are used in string_art_generator.
from image_processing import calculate_line_darkness as actual_calculate_line_darkness
from image_processing import update_image_array as actual_update_image_array
from collections import Counter


# Helper function (similar to one in test_image_processing)
def create_sample_image_array(size=(100, 100), fill_value=255, dtype=np.uint8):
    """Creates a numpy array of a given size, filled with a specific value."""
    return np.full(size, fill_value, dtype=dtype)

def create_mock_pins_coords(num_pins, square_size, radius):
    """Generates pin coordinates using calculate_pins."""
    return calculate_pins(square_size, radius, num_pins)


# Tests for calculate_pins
def test_calculate_pins_zero_pins():
    """Tests calculate_pins with num_pins = 0."""
    pins = calculate_pins(square_size=200, radius=100, num_pins=0)
    assert pins == [], "Expected an empty list for zero pins."

def test_calculate_pins_four_pins():
    """Tests calculate_pins with num_pins = 4 for standard positions."""
    square_size = 200
    radius = 90
    num_pins = 4
    center = square_size / 2 # 100

    pins = calculate_pins(square_size, radius, num_pins)

    assert len(pins) == num_pins, f"Expected {num_pins} pins, got {len(pins)}."

    # Expected coordinates for 4 pins (angles 0, pi/2, pi, 3pi/2)
    # Pin 0 (angle 0): (center + radius*cos(0), center + radius*sin(0)) = (100 + 90, 100 + 0) = (190, 100)
    # Pin 1 (angle pi/2): (center + radius*cos(pi/2), center + radius*sin(pi/2)) = (100 + 0, 100 + 90) = (100, 190)
    # Pin 2 (angle pi): (center + radius*cos(pi), center + radius*sin(pi)) = (100 - 90, 100 + 0) = (10, 100)
    # Pin 3 (angle 3pi/2): (center + radius*cos(3pi/2), center + radius*sin(3pi/2)) = (100 + 0, 100 - 90) = (100, 10)

    expected_pins = [
        (center + radius * math.cos(0), center + radius * math.sin(0)),
        (center + radius * math.cos(math.pi / 2), center + radius * math.sin(math.pi / 2)),
        (center + radius * math.cos(math.pi), center + radius * math.sin(math.pi)),
        (center + radius * math.cos(3 * math.pi / 2), center + radius * math.sin(3 * math.pi / 2)),
    ]

    assert pins[0] == pytest.approx(expected_pins[0]), f"Pin 0 coordinates incorrect. Expected {expected_pins[0]}"
    assert pins[1] == pytest.approx(expected_pins[1]), f"Pin 1 coordinates incorrect. Expected {expected_pins[1]}"
    assert pins[2] == pytest.approx(expected_pins[2]), f"Pin 2 coordinates incorrect. Expected {expected_pins[2]}"
    assert pins[3] == pytest.approx(expected_pins[3]), f"Pin 3 coordinates incorrect. Expected {expected_pins[3]}"

# Tests for to_real_coordinates
def test_to_real_coordinates_empty():
    """Tests to_real_coordinates with an empty list of pins."""
    real_pins = to_real_coordinates([], square_size=200, radius_pixel=100, radius_milimeter=50)
    assert real_pins == [], "Expected an empty list for empty input pins."

def test_to_real_coordinates_simple_conversion():
    """Tests a simple conversion of pixel coordinates to real-world millimeter coordinates."""
    pins_pixel = [(190, 100), (100, 190)] # Using coordinates from the four_pins test
    square_size = 200
    radius_pixel = 90
    radius_milimeter = 45
    center_pixel = square_size / 2 # 100

    # Expected real coordinates:
    # For (190, 100):
    #   x_centered_pixel = 190 - 100 = 90
    #   y_centered_pixel = 100 - 100 = 0
    #   x_real = 45 * 90 / 90 = 45
    #   y_real = 45 * 0 / 90 = 0. So, (45, 0)
    # For (100, 190):
    #   x_centered_pixel = 100 - 100 = 0
    #   y_centered_pixel = 190 - 100 = 90
    #   x_real = 45 * 0 / 90 = 0
    #   y_real = 45 * 90 / 90 = 45. So, (0, 45)
    expected_real_pins = [(45.0, 0.0), (0.0, 45.0)]

    real_pins = to_real_coordinates(pins_pixel, square_size, radius_pixel, radius_milimeter)

    assert len(real_pins) == len(expected_real_pins)
    assert real_pins[0] == pytest.approx(expected_real_pins[0]), f"Real pin 0 incorrect. Expected {expected_real_pins[0]}"
    assert real_pins[1] == pytest.approx(expected_real_pins[1]), f"Real pin 1 incorrect. Expected {expected_real_pins[1]}"

def test_to_real_coordinates_zero_pixel_radius():
    """Tests to_real_coordinates when radius_pixel is 0, expecting (0,0) outputs due to division by zero prevention."""
    pins_pixel = [(190, 100), (10, 100)]
    square_size = 200
    radius_pixel = 0
    radius_milimeter = 50

    # The function should handle division by zero if radius_pixel is 0, typically returning (0,0)
    expected_real_pins = [(0.0, 0.0), (0.0, 0.0)]
    real_pins = to_real_coordinates(pins_pixel, square_size, radius_pixel, radius_milimeter)

    assert len(real_pins) == len(expected_real_pins)
    assert real_pins[0] == pytest.approx(expected_real_pins[0]), "Expected (0,0) for first pin when radius_pixel is 0."
    assert real_pins[1] == pytest.approx(expected_real_pins[1]), "Expected (0,0) for second pin when radius_pixel is 0."


# Tests for find_best_next_pin

# Mock for calculate_line_darkness - to be used with mocker.patch
# It needs to be defined globally or within the test and passed to side_effect.
def mock_line_darkness_factory(darkness_map, default_darkness=200, default_line_len=20):
    """
    Factory to create a mock calculate_line_darkness function.
    darkness_map is a dict of ((pin_idx1, pin_idx2): (darkness, length))
    """
    def mock_calculate_darkness(img_array, x1, y1, x2, y2, pin_coords_map_for_mock=None):
        # Try to find which pins these coordinates correspond to.
        # This is a bit complex if we only have raw coords.
        # Simpler: Assume the test setup maps specific pin_idx pairs to specific darkness/length.
        # The test will need to ensure find_best_next_pin calls this mock with identifiable args
        # OR the mock is structured based on which pins are being connected.
        # For now, this mock will be used with a 'pin_coords_map_for_mock' that maps
        # (x1,y1,x2,y2) back to pin indices if needed, or the test directly uses pin indices.

        # Simplified: The test using this mock should provide a way to identify the pin pair.
        # We'll assume the test using this mock will patch 'calculate_line_darkness'
        # within string_art_generator.py, and the mock will be called from there.
        # The mock needs to differentiate calls. A simple way for tests is to rely on order
        # or specific coordinates if pins are fixed.

        # Let's make the mock expect pin indices directly for easier test setup.
        # This means the test will have to translate this if the actual function takes coords.
        # **This approach is flawed as calculate_line_darkness takes coords, not indices.**
        #
        # Alternative: the mock is independent of indices and just returns values from a list,
        # or the test directly sets up the scenario.

        # For find_best_next_pin, it iterates through potential_next_indices.
        # We can make the mock return specific values based on (current_pin_index, potential_next_index)
        # This means the mock needs to be aware of the context of the call.
        # This is usually done by having the test set up the side_effect function with that context.

        # For this generic factory, let's assume the test will provide a 'current_pin_pair' in the scope
        # or the mock will be replaced per test. This factory is more a template.
        key = "some_key_based_on_args_or_context" # This needs to be determined by the test
        if key in darkness_map:
            darkness, length = darkness_map[key]
            return darkness, list(range(length)) # dummy line pixels
        return default_darkness, list(range(default_line_len)) # default for non-specified lines

    # This factory isn't directly usable with side_effect unless the key can be derived
    # from (img_array, x1, y1, x2, y2).
    # It's better to define specific mocks inside each test or use a more direct patching.
    pass # Placeholder for now. Will define specific mocks in tests.


def test_find_best_next_pin_finds_optimal(mocker):
    """Tests that find_best_next_pin selects the pin on the darkest path."""
    img_size = 100
    init_array = create_sample_image_array((img_size, img_size), 200) # Mostly light
    pins_coords = create_mock_pins_coords(num_pins=4, square_size=img_size, radius=40)
    # Pin 0: (90,50), Pin 1: (50,90), Pin 2: (10,50), Pin 3: (50,10) approx

    # Create a dark path between pin 0 and pin 2.
    # For simplicity, we'll mock calculate_line_darkness.
    def specific_mock_darkness(img_array, x1, y1, x2, y2):
        # Identify pins by coordinates (this is fragile, direct index mocking is better if possible)
        current_pin_coord = (x1,y1)
        next_pin_coord = (x2,y2)
        dark_line_len = 30
        # From Pin 0 to Pin 2
        if current_pin_coord == pytest.approx(pins_coords[0]) and next_pin_coord == pytest.approx(pins_coords[2]):
            return 50.0, [(i,i) for i in range(dark_line_len)] # Dark, valid line
        # From Pin 0 to Pin 1 or Pin 3
        elif current_pin_coord == pytest.approx(pins_coords[0]) and \
             (next_pin_coord == pytest.approx(pins_coords[1]) or next_pin_coord == pytest.approx(pins_coords[3])):
            return 150.0, [(i,i) for i in range(dark_line_len)] # Lighter
        return 200.0, [(i,i) for i in range(dark_line_len)] # Default other lines

    mocker.patch('string_art_generator.calculate_line_darkness', side_effect=specific_mock_darkness)

    current_pin_index = 0
    last_pin_index = None # Or an index not 0, 1, 2, 3 e.g. -1
    current_steps_indices = [0] # Represents how many times each pin has been used.
                                # This needs to be a list of actual pin indices used in sequence.

    pin_usage_counts = Counter(current_steps_indices)


    best_pin_index, best_line_pixels, _ = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        pin_usage_counts, minimum_line_spans=1, max_pin_usage=3
    )

    assert best_pin_index == 2, "Should choose pin 2 for the darkest path."
    assert best_line_pixels is not None
    assert len(best_line_pixels) == 30


def test_find_best_next_pin_respects_max_pin_usage(mocker):
    """Tests that pins exceeding max_pin_usage are not chosen."""
    img_size = 100
    init_array = create_sample_image_array((img_size, img_size), 100) # All moderately dark
    pins_coords = create_mock_pins_coords(num_pins=4, square_size=img_size, radius=40)
    # Pin 0, Pin 1 (darkest), Pin 2, Pin 3

    # Mock calculate_line_darkness: Pin 1 is optimal, Pin 2 is suboptimal but valid
    def mock_darkness(img_array, x1,y1,x2,y2):
        # Robustly find the start pin index by comparing approximated coordinates
        start_pin = -1
        for idx, coord in enumerate(pins_coords):
            if (x1,y1) == pytest.approx(coord, abs=1.0): # Corrected comparison
                start_pin = idx
                break
        if start_pin == -1: # Should not happen if test is set up correctly
            raise ValueError(f"Mock 'mock_darkness' could not identify start pin for int coords ({x1},{y1}) with float pins {pins_coords}")

        # Compare end_pin_approx with approximated pin coordinates
        if start_pin == 0 and ((x2,y2) == pytest.approx(pins_coords[1], abs=1.0)): # Path 0 -> 1
            return 50.0, [(i,i) for i in range(20)] # Darkest
        if start_pin == 0 and ((x2,y2) == pytest.approx(pins_coords[2], abs=1.0)): # Path 0 -> 2
            return 100.0, [(i,i) for i in range(20)] # Suboptimal
        if start_pin == 0 and ((x2,y2) == pytest.approx(pins_coords[3], abs=1.0)): # Path 0 -> 3
             return 150.0, [(i,i) for i in range(20)] # Lightest
        # Fallback for any other pins from start_pin 0
        if start_pin == 0:
            return 200.0, [(i,i) for i in range(20)]
        return 250.0, [(i,i) for i in range(20)] # Default for any other start_pin


    mocker.patch('string_art_generator.calculate_line_darkness', side_effect=mock_darkness)

    current_pin_index = 0
    last_pin_index = -1 # Not relevant here

    # Pin 1 used 3 times (max_pin_usage)
    # Pin 2 used 1 time
    # Pin 3 used 1 time
    pin_usage_counts = Counter({0:1, 1:3, 2:1, 3:1})
    max_pin_usage = 3

    best_pin_index, _, _ = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        pin_usage_counts, minimum_line_spans=1, max_pin_usage=max_pin_usage
    )

    assert best_pin_index == 2, "Should choose pin 2 as pin 1 is maxed out."

    # Scenario: Pin 1 is the only option, but it's maxed out
    pin_usage_counts_all_maxed_except_one_target = Counter({0:1, 1:3, 2:3, 3:3})
    # And all lines to other pins are e.g. too short or non-existent (mocked high darkness)
    def mock_darkness_only_one_good_but_maxed(img_array, x1,y1,x2,y2):
        start_pin = -1
        for idx, coord in enumerate(pins_coords):
            if (x1,y1) == pytest.approx(coord, abs=1.0): # Corrected comparison
                start_pin = idx
                break
        if start_pin == -1: raise ValueError(f"Mock could not identify start pin for int coords ({x1},{y1}) with float pins {pins_coords}")

        if start_pin == 0 and ((x2,y2) == pytest.approx(pins_coords[1], abs=1.0)): # Path 0 -> 1 (darkest)
            return 50.0, [(i,i) for i in range(20)]
        # Fallback for any other pins from start_pin 0
        if start_pin == 0:
            return 255.0, []
        return 255.0, [] # Default for any other start_pin

    mocker.patch('string_art_generator.calculate_line_darkness', side_effect=mock_darkness_only_one_good_but_maxed)

    best_pin_index_none, _, _ = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        pin_usage_counts_all_maxed_except_one_target, # Pin 1 (target) is maxed
        minimum_line_spans=1, max_pin_usage=max_pin_usage
    )
    assert best_pin_index_none is None, "Should return None if the only viable pin is maxed out."


def test_find_best_next_pin_respects_minimum_pins_line_spans(mocker):
    """Tests that lines that do not span enough pins are ignored."""
    img_size = 100
    init_array = create_sample_image_array((img_size, img_size), 100) # All moderately dark
    pins_coords = create_mock_pins_coords(num_pins=3, square_size=img_size, radius=40)
    # Pin 0, Pin 1 (darkest but too close), Pin 2 (suboptimal but far enough)

    minimum_line_spans = 2

    def mock_darkness(img_array, x1,y1,x2,y2):
        start_pin = -1
        for idx, coord in enumerate(pins_coords):
            if (x1,y1) == pytest.approx(coord, abs=1.0):
                start_pin = idx
                break
        if start_pin == -1: raise ValueError(f"Mock could not identify start pin for int coords ({x1},{y1}) with float pins {pins_coords}")

        if start_pin == 0 and ((x2,y2) == pytest.approx(pins_coords[1], abs=1.0)): # Path 0 -> 1 (adjacent, not enough span)
            return 50.0, [(i,i) for i in range(20)] # Darkest, but not enough pins spanned
        if start_pin == 0 and ((x2,y2) == pytest.approx(pins_coords[2], abs=1.0)): # Path 0 -> 2 (spans 2 pins)
            return 100.0, [(i,i) for i in range(25)] # Suboptimal, but enough pins spanned
        if start_pin == 0:
            return 200.0, [(i,i) for i in range(20)]
        return 250.0, [(i,i) for i in range(20)]

    mocker.patch('string_art_generator.calculate_line_darkness', side_effect=mock_darkness)

    current_pin_index = 0
    last_pin_index = -1
    pin_usage_counts = Counter({0:1})

    best_pin_index, _, _ = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        pin_usage_counts, minimum_line_spans=minimum_line_spans, max_pin_usage=3
    )

    assert best_pin_index == 2, "Should choose pin 2 as pin 1's line does not span enough pins."

    # Scenario: All lines do not span enough pins
    def mock_darkness_all_too_close(img_array, x1,y1,x2,y2):
        return 50.0, [(i,i) for i in range(20)]
    mocker.patch('string_art_generator.calculate_line_darkness', side_effect=mock_darkness_all_too_close)
    # Set minimum_line_spans to a value greater than the max possible index difference (which is 2 for 3 pins)
    best_pin_index_none, _, _ = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        pin_usage_counts, minimum_line_spans=3, max_pin_usage=3
    )
    assert best_pin_index_none is None, "Should return None if all lines do not span enough pins."


def test_find_best_next_pin_avoids_last_pin(mocker):
    """Tests that find_best_next_pin avoids going directly back to last_pin_index."""
    img_size = 100
    init_array = create_sample_image_array((img_size, img_size), 100)
    pins_coords = create_mock_pins_coords(num_pins=3, square_size=img_size, radius=40)
    # Pin 0, Pin 1, Pin 2

    current_pin_index = 1
    last_pin_index = 0 # Should not go back to pin 0 from pin 1

    def mock_darkness(img_array, x1,y1,x2,y2):
        start_pin = -1
        for idx, coord in enumerate(pins_coords):
            if (x1,y1) == pytest.approx(coord, abs=1.0): # Corrected comparison
                start_pin = idx
                break
        if start_pin == -1: raise ValueError(f"Mock could not identify start pin for int coords ({x1},{y1}) with float pins {pins_coords}")

        if start_pin == 1 and ((x2,y2) == pytest.approx(pins_coords[0], abs=1.0)): # Path 1 -> 0 (last_pin_index)
            return 10.0, [(i,i) for i in range(20)] # Very dark, but should be avoided
        if start_pin == 1 and ((x2,y2) == pytest.approx(pins_coords[2], abs=1.0)): # Path 1 -> 2
            return 80.0, [(i,i) for i in range(20)] # Moderately dark, valid choice
        # Fallback for any other pins from start_pin 1
        if start_pin == 1:
            return 200.0, [(i,i) for i in range(20)]
        return 250.0, [(i,i) for i in range(20)] # Default for any other start_pin

    mocker.patch('string_art_generator.calculate_line_darkness', side_effect=mock_darkness)

    pin_usage_counts = Counter({0:1, 1:1})
    best_pin_index, _, _ = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        pin_usage_counts, minimum_line_spans=1, max_pin_usage=3
    )

    assert best_pin_index == 2, "Should choose pin 2, avoiding going back to last_pin_index 0."


def test_find_best_next_pin_no_suitable_pin(mocker):
    """Tests scenario where no pin meets the criteria (e.g., image is all white)."""
    img_size = 100
    init_array = create_sample_image_array((img_size, img_size), 255) # All white
    pins_coords = create_mock_pins_coords(num_pins=4, square_size=img_size, radius=40)

    # No need to mock calculate_line_darkness if actual function is used,
    # as it will correctly return high darkness values for a white image.
    # If we want to be super sure or test specific edge cases of "no suitable pin":
    mocker.patch('string_art_generator.calculate_line_darkness', return_value=(255.0, [])) # All lines white, empty pixels

    current_pin_index = 0
    last_pin_index = -1
    pin_usage_counts = Counter({0:1})

    best_pin_index, best_line_pixels, _ = find_best_next_pin(
        pins_coords, init_array, current_pin_index, last_pin_index,
        pin_usage_counts, minimum_line_spans=1, max_pin_usage=3
    )

    assert best_pin_index is None, "Expected no suitable pin if image is all white or lines are too light."
    assert best_line_pixels is None


# Tests for generate_next_string_segment
# We need to mock string_art_generator.update_image_array and string_art_generator.find_best_next_pin for these.

@pytest.fixture
def mock_widgets_basic():
    img_size = 50 # Smaller for faster tests
    pins_coords = create_mock_pins_coords(num_pins=4, square_size=img_size, radius=20)
    init_array = create_sample_image_array((img_size,img_size), 100) # moderately dark image
    return {
        "pins": pins_coords,
        "current_index": 0,
        "last_index": -1, # Using -1 to indicate no previous pin for simplicity
        "init_array": init_array,
        "parameters": {"Minimum Pins Line Spans": 1, "Max Pin Usage": 3},
        "steps_index": [0], # Initial state, current_index is 0
        "pin_usage": Counter([0]) # Initial state
    }

def test_generate_next_string_segment_success(mocker, mock_widgets_basic):
    """Tests successful generation of a string segment."""
    widgets = mock_widgets_basic

    # Mock find_best_next_pin to return a valid next pin
    # Let's say pin 2 is chosen, line from (0,0) to (1,1) for simplicity
    mock_best_pin_index = 2
    mock_best_line_pixels = [(0,0), (1,1)]
    mock_best_darkness = 50.0
    mocker.patch('string_art_generator.find_best_next_pin',
                 return_value=(mock_best_pin_index, mock_best_line_pixels, mock_best_darkness))

    # Mock update_image_array as it's called internally
    mock_update_image = mocker.patch('string_art_generator.update_image_array')

    next_pin_coord, line_pixel_coords, next_pin_index = generate_next_string_segment(widgets)

    assert next_pin_coord == widgets["pins"][mock_best_pin_index]
    assert line_pixel_coords == mock_best_line_pixels
    assert next_pin_index == mock_best_pin_index
    mock_update_image.assert_called_once_with(widgets, mock_best_line_pixels)


def test_generate_next_string_segment_failure_no_pin_found(mocker, mock_widgets_basic):
    """Tests failure case where find_best_next_pin finds no suitable pin."""
    widgets = mock_widgets_basic

    # Mock find_best_next_pin to return None (no pin found)
    mocker.patch('string_art_generator.find_best_next_pin', return_value=(None, None, None))
    mock_update_image = mocker.patch('string_art_generator.update_image_array')


    next_pin_coord, line_pixel_coords, next_pin_index = generate_next_string_segment(widgets)

    assert next_pin_coord is None
    assert line_pixel_coords is None
    assert next_pin_index is None
    mock_update_image.assert_not_called()
