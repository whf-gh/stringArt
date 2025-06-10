import numpy as np
from PIL import Image
import pytest # It's good practice to import pytest if we're using its features, though not strictly needed for basic assert
import sys
import os

# Add the project root to the Python path to allow importing image_processing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from image_processing import (
    calculate_line_darkness,
    apply_circle_mask,
    process_image,
    select_image,
    update_image_array
    # Note: update_image_array might be harder to test in isolation
    # as it modifies a 'widgets' dictionary structure.
)

# Helper function to create a sample image array
def create_sample_image_array(size=(10, 10), fill_value=128, dtype=np.uint8):
    """Creates a numpy array of a given size, filled with a specific value."""
    return np.full(size, fill_value, dtype=dtype)

# Test function for calculate_line_darkness
def test_calculate_line_darkness_simple_horizontal_line():
    """Tests calculate_line_darkness with a simple horizontal line on a uniform image."""
    img_array = create_sample_image_array(size=(10, 10), fill_value=100)

    # Horizontal line from (x1, y1) = (1, 5) to (x2, y2) = (8, 5)
    # Expected pixels in line: (1,5), (2,5), (3,5), (4,5), (5,5), (6,5), (7,5), (8,5) - 8 pixels
    # skimage.draw.line includes endpoints.
    x1, y1 = 1, 5
    x2, y2 = 8, 5

    darkness, line_pixels = calculate_line_darkness(img_array, x1, y1, x2, y2)

    # Assertions
    assert np.isclose(darkness, 100.0), f"Expected darkness ~100.0, got {darkness}"

    expected_line_coords = []
    for x in range(x1, x2 + 1): # x iterates from 1 to 8
        expected_line_coords.append((x,y1)) # (x,y) format

    assert len(line_pixels) == len(expected_line_coords), \
        f"Expected {len(expected_line_coords)} pixels, got {len(line_pixels)}"

    # Convert to sets for order-independent comparison, though order should be preserved by skimage.draw.line
    assert set(line_pixels) == set(expected_line_coords), \
        f"Line pixel coordinates do not match. Expected: {expected_line_coords}, Got: {line_pixels}"


def test_calculate_line_darkness_vertical_line():
    """Tests calculate_line_darkness with a simple vertical line on a uniform image."""
    img_array = create_sample_image_array(size=(10, 10), fill_value=50)
    x1, y1 = 5, 1
    x2, y2 = 5, 8 # Vertical line from (5,1) to (5,8)

    darkness, line_pixels = calculate_line_darkness(img_array, x1, y1, x2, y2)

    assert np.isclose(darkness, 50.0), f"Expected darkness ~50.0, got {darkness}"

    expected_line_coords = []
    for y in range(y1, y2 + 1):
        expected_line_coords.append((x1, y))

    assert len(line_pixels) == len(expected_line_coords), \
        f"Expected {len(expected_line_coords)} pixels, got {len(line_pixels)}"
    assert set(line_pixels) == set(expected_line_coords), \
        f"Line pixel coordinates do not match. Expected: {expected_line_coords}, Got: {line_pixels}"


def test_calculate_line_darkness_diagonal_line():
    """Tests calculate_line_darkness with a simple diagonal line on a uniform image."""
    img_array = create_sample_image_array(size=(10, 10), fill_value=75)
    # Diagonal line from (1,1) to (8,8)
    # skimage.draw.line will generate pixels for this diagonal.
    # For a perfect diagonal like this, it should be (1,1), (2,2), ..., (8,8)
    x1, y1 = 1, 1
    x2, y2 = 8, 8

    darkness, line_pixels = calculate_line_darkness(img_array, x1, y1, x2, y2)

    assert np.isclose(darkness, 75.0), f"Expected darkness ~75.0, got {darkness}"

    expected_line_coords = []
    for i in range(x1, x2 + 1):
        expected_line_coords.append((i,i))

    # skimage.draw.line might produce slightly different results for diagonals depending on implementation
    # For a simple diagonal (1,1) to (8,8) on a grid, it should hit (1,1), (2,2)...(8,8)
    # Checking length and set comparison should be robust.
    assert len(line_pixels) == len(expected_line_coords), \
         f"Expected {len(expected_line_coords)} pixels, got {len(line_pixels)}. Got: {line_pixels}"
    assert set(line_pixels) == set(expected_line_coords), \
        f"Line pixel coordinates do not match. Expected: {expected_line_coords}, Got: {line_pixels}"


def test_calculate_line_darkness_line_out_of_bounds():
    """Tests calculate_line_darkness with a line partially out of bounds."""
    img_array = create_sample_image_array(size=(5, 5), fill_value=100) # Small 5x5 image

    # Line from (1,1) to (6,1) - part of it is outside x-bounds (0-4)
    x1, y1 = 1, 1
    x2, y2 = 6, 1 # x goes up to 6, max index is 4

    darkness, line_pixels = calculate_line_darkness(img_array, x1, y1, x2, y2)

    # Pixels within bounds: (1,1), (2,1), (3,1), (4,1)
    # All these pixels have value 100.
    assert np.isclose(darkness, 100.0), f"Expected darkness ~100.0 for in-bound part, got {darkness}"

    expected_line_coords_in_bounds = [(1,1), (2,1), (3,1), (4,1)]
    assert len(line_pixels) == len(expected_line_coords_in_bounds), \
        f"Expected {len(expected_line_coords_in_bounds)} in-bound pixels, got {len(line_pixels)}. Got: {line_pixels}"
    assert set(line_pixels) == set(expected_line_coords_in_bounds), \
        f"Line pixel coordinates do not match. Expected: {expected_line_coords_in_bounds}, Got: {line_pixels}"

    # Line completely out of bounds
    darkness_out, line_pixels_out = calculate_line_darkness(img_array, 10, 10, 12, 10)
    assert darkness_out == 255.0, f"Expected 255.0 for fully out-of-bounds line, got {darkness_out}"
    assert len(line_pixels_out) == 0, f"Expected 0 pixels for fully out-of-bounds line, got {len(line_pixels_out)}"


def test_calculate_line_darkness_zero_length_line():
    """Tests calculate_line_darkness with a zero-length line (start == end)."""
    img_array = create_sample_image_array(size=(10, 10), fill_value=120)
    x1, y1 = 3, 3

    darkness, line_pixels = calculate_line_darkness(img_array, x1, y1, x1, y1)

    assert np.isclose(darkness, 120.0), f"Expected darkness for a single pixel, got {darkness}"
    assert len(line_pixels) == 1, f"Expected 1 pixel for a zero-length line, got {len(line_pixels)}"
    assert set(line_pixels) == {(x1,y1)}, \
        f"Pixel coordinate incorrect. Expected: {(x1,y1)}, Got: {line_pixels}"


# Tests for update_image_array
DUMMY_IMAGE_PATH = "tests/assets/dummy_image.png" # 10x10 image

def test_update_image_array_simple_update():
    """Tests the basic functionality of update_image_array."""
    init_array = create_sample_image_array(size=(5, 5), fill_value=150)
    widgets = {"init_array": init_array.copy()} # Pass a copy

    # Line pixels to update - e.g., a short horizontal line
    line_to_update = [(1, 2), (2, 2), (3, 2)]

    update_image_array(widgets, line_to_update)

    updated_array = widgets["init_array"]

    # Check that specified pixels are updated.
    # According to logic: init_array[y, x] = min(int(init_array[y, x] + 100), 255)
    # So, 150 + 100 = 250
    expected_value_after_update = 250
    for x, y in line_to_update:
        assert updated_array[y, x] == expected_value_after_update, \
            f"Pixel ({x},{y}) was not updated correctly. Expected {expected_value_after_update}, got {updated_array[y,x]}"

    # Check that other pixels are not changed
    for r in range(updated_array.shape[0]):
        for c in range(updated_array.shape[1]):
            if (c, r) not in line_to_update: # (c,r) is (x,y)
                 assert updated_array[r, c] == 150, \
                    f"Pixel ({c},{r}) was changed unexpectedly. Expected 150, got {updated_array[r,c]}"


def test_update_image_array_with_processed_array():
    """Tests update_image_array logic when processed_array is involved."""
    init_array = create_sample_image_array(size=(5, 5), fill_value=100)
    processed_array = create_sample_image_array(size=(5, 5), fill_value=255) # all white initially

    # Make some pixels in processed_array black (0)
    processed_array[2, 1] = 0 # (x=1, y=2)
    processed_array[2, 2] = 0 # (x=2, y=2)

    widgets = {
        "init_array": init_array.copy(),
        "processed_array": processed_array.copy()
    }

    # Line that overlaps with the black pixels in processed_array
    line_to_update = [(1, 2), (2, 2), (3, 2)] # (x,y)
                                          # (1,2) -> processed_array[2,1] == 0
                                          # (2,2) -> processed_array[2,2] == 0
                                          # (3,2) -> processed_array[2,3] == 255

    update_image_array(widgets, line_to_update)
    updated_init_array = widgets["init_array"]

    # Check pixel (1,2) (y=2, x=1): init_array[2,1] was 100. processed_array[2,1] is 0.
    # Logic: init_array[y,x] = init_array[y,x] + (255 - init_array[y,x]) // 2
    # 100 + (255 - 100)//2 = 100 + 155//2 = 100 + 77 = 177
    assert updated_init_array[2, 1] == 177, f"Pixel (1,2) updated incorrectly with processed_array. Expected 177, got {updated_init_array[2,1]}"

    # Check pixel (2,2) (y=2, x=2): init_array[2,2] was 100. processed_array[2,2] is 0.
    assert updated_init_array[2, 2] == 177, f"Pixel (2,2) updated incorrectly with processed_array. Expected 177, got {updated_init_array[2,2]}"

    # Check pixel (3,2) (y=2, x=3): init_array[2,3] was 100. processed_array[2,3] is 255.
    # Logic: init_array[y,x] = 255
    assert updated_init_array[2, 3] == 255, f"Pixel (3,2) updated incorrectly (should be whitened). Expected 255, got {updated_init_array[2,3]}"


# Tests for apply_circle_mask
def test_apply_circle_mask_basic():
    """Tests the basic application of a circular mask."""
    img_size = 50
    radius = 20
    center_x, center_y = img_size // 2, img_size // 2 # 25, 25

    # Create a solid blue image
    original_image = Image.new("RGB", (img_size, img_size), color="blue")

    masked_image = apply_circle_mask(original_image, (center_x, center_y), radius)

    assert isinstance(masked_image, Image.Image), "apply_circle_mask should return a PIL Image object."
    assert masked_image.mode == "RGB", f"Masked image mode should be RGB, got {masked_image.mode}"
    assert masked_image.size == (img_size, img_size), "Masked image size should match original."

    # Check pixel colors
    # Pixel inside the circle (e.g., center) should be blue
    # Pixel far outside the circle (e.g., corner) should be white

    # Center pixel (25,25) should be blue (0,0,255)
    assert masked_image.getpixel((center_x, center_y)) == (0, 0, 255), \
        f"Center pixel color incorrect. Expected blue, got {masked_image.getpixel((center_x, center_y))}"

    # Corner pixel (0,0) should be white (255,255,255)
    assert masked_image.getpixel((0, 0)) == (255, 255, 255), \
        f"Corner pixel color incorrect. Expected white, got {masked_image.getpixel((0,0))}"

    # A pixel just inside the edge of the circle, e.g. (center_x + radius -1, center_y)
    # (25 + 20 - 1, 25) = (44, 25)
    dist_sq = ( (center_x + radius -1) - center_x)**2 + (center_y - center_y)**2
    if dist_sq < radius**2: # If truly inside
         assert masked_image.getpixel((center_x + radius - 1, center_y)) == (0,0,255), \
             "Pixel just inside circle edge should be blue"

    # A pixel just outside the edge, e.g. (center_x + radius + 1, center_y)
    # (25 + 20 + 1, 25) = (46,25) - this might be out of bounds if radius is close to img_size/2
    # Let's use a point guaranteed to be outside: (center_x, 0) if radius < center_y
    if center_y - radius > 0: # ensure (center_x, 0) is outside
        assert masked_image.getpixel((center_x, 0)) == (255, 255, 255), \
            "Pixel clearly outside circle (top edge center) should be white"


# Tests for select_image
# Note: select_image involves Tkinter for dialogs if test_file_path is not provided.
# We are testing the case where test_file_path IS provided.

def test_select_image_with_test_path():
    """Tests select_image using the test_file_path argument."""
    # dummy_image.png is 10x10.
    # If image_square_size is 20, it should scale up.
    # Then it will be cropped to 20x20 (no change if already square after scaling).
    target_size = 20

    # Ensure the dummy image exists
    if not os.path.exists(DUMMY_IMAGE_PATH):
        pytest.fail(f"Test asset not found: {DUMMY_IMAGE_PATH}")

    img, img_original = select_image(image_square_size=target_size, test_file_path=DUMMY_IMAGE_PATH)

    assert isinstance(img, Image.Image), "Returned 'image' should be a PIL Image."
    assert isinstance(img_original, Image.Image), "Returned 'image_original' should be a PIL Image."

    assert img.size == (target_size, target_size), \
        f"Processed image size incorrect. Expected ({target_size},{target_size}), got {img.size}"

    # image_original should be a copy of the processed image in this function's current implementation
    assert img_original.size == (target_size, target_size), \
        f"Original image (copy) size incorrect. Expected ({target_size},{target_size}), got {img_original.size}"
    # Further check: img_original should be a copy of the *loaded and processed* image, not the raw original file.
    # The dummy image is 10x10. select_image scales it to 20x20. So img_original should also be 20x20.


def test_select_image_resizing_and_cropping():
    """Tests select_image's resizing and cropping logic."""
    # dummy_image.png is 10x10.
    if not os.path.exists(DUMMY_IMAGE_PATH):
        pytest.fail(f"Test asset not found: {DUMMY_IMAGE_PATH}")

    # Case 1: Upscale and then crop (if it became non-square, though scaling maintains aspect)
    # If image_square_size = 15. Original is 10x10.
    # Scale factor = max(15/10, 15/10) = 1.5. New size = (15, 15).
    # Then crop: left=(15-15)/2=0, top=0. right=15, bottom=15. Crop is (0,0,15,15). Result is 15x15.
    target_size_upscale = 15
    img_up, _ = select_image(image_square_size=target_size_upscale, test_file_path=DUMMY_IMAGE_PATH)
    assert img_up.size == (target_size_upscale, target_size_upscale), \
        f"Upscaled image size: Expected ({target_size_upscale},{target_size_upscale}), got {img_up.size}"

    # Case 2: Downscale (if original was larger) - dummy is 10x10, so this won't downscale.
    # Instead, test cropping behavior when image_square_size is smaller than one dimension after potential scaling.
    # Create a temporary non-square image for this test.
    temp_img_path = "tests/assets/temp_non_square.png"
    try:
        non_square_pil = Image.new("L", (30, 20), color=128) # 30 wide, 20 tall
        non_square_pil.save(temp_img_path)

        # image_square_size = 18. Original is 30x20.
        # No upscaling needed as 30 > 18 and 20 > 18.
        # Crop:
        # left = (30-18)/2 = 6
        # top = (20-18)/2 = 1
        # right = 6 + 18 = 24
        # bottom = 1 + 18 = 19
        # Resulting crop area: (6, 1, 24, 19), which is 18x18.
        target_size_crop = 18
        img_crop, _ = select_image(image_square_size=target_size_crop, test_file_path=temp_img_path)
        assert img_crop.size == (target_size_crop, target_size_crop), \
            f"Cropped image size: Expected ({target_size_crop},{target_size_crop}), got {img_crop.size}"
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    # Case 3: image_square_size smaller than dummy image (10x10), e.g. 8x8
    # Original 10x10. image_square_size = 8.
    # No upscaling as 10 > 8.
    # Crop: left=(10-8)/2=1, top=1, right=1+8=9, bottom=1+8=9. Crop is (1,1,9,9). Result is 8x8.
    target_size_down_crop = 8
    img_down_crop, _ = select_image(image_square_size=target_size_down_crop, test_file_path=DUMMY_IMAGE_PATH)
    assert img_down_crop.size == (target_size_down_crop, target_size_down_crop), \
        f"Down_cropped image size: Expected ({target_size_down_crop},{target_size_down_crop}), got {img_down_crop.size}"


# Tests for process_image
@pytest.fixture
def dummy_pil_image():
    """Loads the dummy_image.png as a PIL Image for process_image tests."""
    if not os.path.exists(DUMMY_IMAGE_PATH):
        pytest.fail(f"Test asset not found: {DUMMY_IMAGE_PATH}. Run generation script.")
    return Image.open(DUMMY_IMAGE_PATH)

def test_process_image_no_operations(dummy_pil_image):
    """Tests process_image with no processing options selected."""
    image_to_process = dummy_pil_image.copy() # dummy_image is 10x10
    checkboxes_state = {
        "Remove Background": False, "denoise": False, "Canny edge detection": False,
        "Adaptive thresholding": False, "Global Thresholding": False,
        "Line Thickness": False, "Use Processed Image Only": False
    }
    image_square_size = 10 # Keep original size for simplicity here
    radius_pixels = 5      # Circle mask radius

    # Convert initial PIL image to array (as it would be before processing starts)
    # process_image expects the 'image_to_process' to be the original selected image (like widgets["image_original"])
    # and 'existing_init_array' to be an array version of it (like widgets["init_array"] before this processing call)
    existing_init_array = np.array(image_to_process.convert("L"))

    final_pil_image, init_array_to_use, output_processed_array = process_image(
        image_to_process, checkboxes_state, image_square_size, radius_pixels, existing_init_array
    )

    assert isinstance(final_pil_image, Image.Image), "final_pil_image should be a PIL Image."
    assert final_pil_image.size == (image_square_size, image_square_size)
    assert isinstance(init_array_to_use, np.ndarray), "init_array_to_use should be a numpy array."
    assert init_array_to_use.shape == (image_square_size, image_square_size)
    assert isinstance(output_processed_array, np.ndarray), "output_processed_array should be a numpy array."
    assert output_processed_array.shape == (image_square_size, image_square_size)

    # Since "Use Processed Image Only" is False, init_array_to_use should be the masked original.
    # And since no processing ops are done, final_pil_image (and thus output_processed_array)
    # should also be just the masked original.

    # Create expected masked original array
    expected_masked_original_pil = apply_circle_mask(
        image_to_process.convert("L"),
        (image_square_size // 2, image_square_size // 2),
        radius_pixels
    )
    expected_masked_original_array = np.array(expected_masked_original_pil.convert("L"))

    assert np.array_equal(init_array_to_use, expected_masked_original_array), \
        "init_array_to_use should be the masked original when no ops and not 'Use Processed Only'"

    assert np.array_equal(output_processed_array, expected_masked_original_array), \
        "output_processed_array should be the masked original when no actual processing ops are done"

def test_process_image_remove_background(dummy_pil_image):
    """Tests process_image with 'Remove Background' enabled."""
    image_to_process = dummy_pil_image.copy() # dummy_image is 10x10 with a black square on white
    checkboxes_state = {"Remove Background": True} # All others false by default if not mentioned
    image_square_size = 10
    radius_pixels = 5
    existing_init_array = np.array(image_to_process.convert("L"))

    final_pil_image, init_array_to_use, output_processed_array = process_image(
        image_to_process, checkboxes_state, image_square_size, radius_pixels, existing_init_array
    )

    assert isinstance(final_pil_image, Image.Image)
    assert final_pil_image.size == (image_square_size, image_square_size)
    assert isinstance(init_array_to_use, np.ndarray)
    assert init_array_to_use.shape == (image_square_size, image_square_size)
    assert isinstance(output_processed_array, np.ndarray) # Will be the bg-removed, then masked image
    assert output_processed_array.shape == (image_square_size, image_square_size)

    # It's hard to check exact pixel values of rembg output without a reference image.
    # For now, just check that it ran and output types/shapes are okay.
    # A simple check: our dummy image has a black square. Background removal might alter it.
    # The original dummy image has a 3x3 black square (value 0) at center on white (255).
    # After rembg, if background is removed, the white parts might become transparent then white again.
    # The black square might remain.
    # A very basic check: the mean of output_processed_array should not be entirely 255 (all white).
    assert np.mean(output_processed_array) < 250, "Image after bg removal and mask shouldn't be all white."


def test_process_image_canny_edge(dummy_pil_image):
    """Tests process_image with 'Canny edge detection' enabled."""
    image_to_process = dummy_pil_image.copy()
    checkboxes_state = {"Canny edge detection": True}
    image_square_size = 10
    radius_pixels = 5
    existing_init_array = np.array(image_to_process.convert("L"))

    final_pil_image, init_array_to_use, output_processed_array = process_image(
        image_to_process, checkboxes_state, image_square_size, radius_pixels, existing_init_array
    )

    assert isinstance(final_pil_image, Image.Image)
    assert final_pil_image.size == (image_square_size, image_square_size)
    assert isinstance(output_processed_array, np.ndarray)
    assert output_processed_array.shape == (image_square_size, image_square_size)

    # Canny output should be mostly black (0) with white (255) edges.
    # Check if there are some edge pixels.
    # Count non-zero (white) pixels. Should be > 0 for an image with edges.
    # The dummy image has a square, so it has clear edges.
    edge_pixels_count = np.sum(output_processed_array == 255)
    assert edge_pixels_count > 0, "Canny edge should detect some edges in the dummy image."

    # Also, the mean value of a Canny image is usually low (many black pixels).
    assert np.mean(output_processed_array) < 100, \
        "Mean of Canny edge output should be relatively low." # Arbitrary threshold, but much less than 255

    # init_array_to_use should be the masked original image here
    expected_masked_original_pil = apply_circle_mask(
        image_to_process.convert("L"),
        (image_square_size // 2, image_square_size // 2),
        radius_pixels
    )
    expected_masked_original_array = np.array(expected_masked_original_pil.convert("L"))
    assert np.array_equal(init_array_to_use, expected_masked_original_array), \
        "init_array_to_use should be the masked original when Canny is on but not 'Use Processed Only'"


def test_process_image_use_processed_only(dummy_pil_image):
    """Tests 'Use Processed Image Only' correctly sets init_array_to_use and output_processed_array."""
    image_to_process = dummy_pil_image.copy()
    checkboxes_state = {"Canny edge detection": True, "Use Processed Image Only": True}
    image_square_size = 10
    radius_pixels = 5
    existing_init_array = np.array(image_to_process.convert("L"))

    final_pil_image, init_array_to_use, output_processed_array = process_image(
        image_to_process, checkboxes_state, image_square_size, radius_pixels, existing_init_array
    )

    assert isinstance(final_pil_image, Image.Image) # This is the PIL version of output_processed_array from canny
    assert final_pil_image.size == (image_square_size, image_square_size)

    # final_pil_image is from the processed array (Canny), then masked.
    # init_array_to_use should be the same as this.
    expected_init_array = np.array(final_pil_image.convert("L"))
    assert np.array_equal(init_array_to_use, expected_init_array), \
        "init_array_to_use should be the Canny output (masked) when 'Use Processed Image Only' is True."

    assert output_processed_array is None, \
        "output_processed_array should be None when 'Use Processed Image Only' is True."
