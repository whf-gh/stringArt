import cv2
import copy
import math
import numpy as np
import pygame as pg
from enum import Enum
from rembg import remove
from PIL import Image, ImageDraw
from tkinter import Tk, filedialog
from net_server import TCPServer as Server
import random


class State(Enum):
    CONFIGURING = 1
    DRAWING = 2
    PAUSING = 3
    SERVING = 4
    QUITING = 5


def init_widgets():
    widgets = {}
    pg.init()
    scrsize = np.array(pg.display.get_desktop_sizes()[0], dtype=int) * 0.9
    square_size = int(scrsize[1])
    widgets["window"] = pg.display.set_mode(scrsize)
    pg.display.set_caption("StringArtProcess")
    widgets["image_square_size"] = int(square_size)
    widgets["string_drawing_surface"] = pg.surface.Surface(
        (square_size, square_size), pg.SRCALPHA
    )
    widgets["information_surface"] = pg.surface.Surface(
        (scrsize[0] - square_size, square_size), pg.SRCALPHA
    )
    fontsize = int(scrsize[1] * 0.03)
    widgets["font"] = pg.font.SysFont("Arial", fontsize)
    widgets["color_white"] = (255, 255, 255)
    widgets["color_black"] = (0, 0, 0)
    widgets["parameters"] = {}
    widgets["labels"] = []
    widgets["input_boxes"] = []
    widgets["defaults"] = {}
    widgets["buttons"] = {}
    widgets["button_boxes"] = {}
    widgets["button_label"] = {}
    widgets["checkboxes"] = {}
    widgets["checkbox_boxes"] = {}
    widgets["checkbox_labels"] = {}
    widgets["active_box"] = None
    widgets["cursor_visible"] = True
    widgets["state"] = State.CONFIGURING
    widgets["image"] = None
    widgets["image_original"] = None
    widgets["decay"] = 1.5
    widgets["init_array"] = None
    widgets["processed_array"] = None
    widgets["current_index"] = 0
    widgets["last_index"] = None
    widgets["pins"] = []
    widgets["use_pin_index"] = False
    widgets["steps"] = []
    widgets["steps_index"] = []
    widgets["lines"] = []
    widgets["total_length_in_pixels"] = 0
    widgets["server"] = None
    return widgets


def create_config_widgets(widgets):
    params = {
        "Number of Pins": 200,
        "Number of Lines": 500,
        "Radius in Pixels": widgets["image_square_size"] // 2,
        "Radius in milimeter": 190,
        "Shortest Line in Pixels": widgets["image_square_size"] // 2,
        "Max Pin Usage": 15,
    }
    buttons = {"Select Image": select_image, "Submit": submit_parameters}
    checkboxes = {
        "Remove Background": False,
        "denoise": False,
        "Canny edge detection": False,
        "Adaptive thresholding": False,
        "Global Thresholding": False,
        "Line Thickness": False,
        "Use Processed Image Only": False,
        "Use Pin Index": widgets["use_pin_index"],
    }
    widgets["parameters"] = params
    widgets["buttons"] = buttons
    widgets["checkboxes"] = checkboxes
    margin = widgets["image_square_size"] * 0.05
    position = [margin, margin]
    labal_position = position
    font = widgets["font"]
    labal_length = font.size(max(params.keys(), key=len))[0]
    font_size = font.size("1000000")
    if widgets["image_original"] is not None:
        widgets["image"] = widgets["image_original"].copy()
    for param, default in params.items():
        label = font.render(param, True, (255, 255, 255))
        widgets["labels"].append((label, copy.copy(labal_position)))
        input_box = pg.Rect(
            labal_position[0] + labal_length + font_size[1],
            labal_position[1],
            font_size[0],
            font_size[1] * 1.2,
        )
        widgets["input_boxes"].append(input_box)
        widgets["defaults"][param] = default  # Set default value
        labal_position[1] += font_size[1] * 1.7
    for checkbox_label in checkboxes.keys():
        checkbox_box = pg.Rect(
            labal_position[0], labal_position[1], font_size[1], font_size[1]
        )
        widgets["checkbox_boxes"][checkbox_label] = checkbox_box
        widgets["checkbox_labels"][checkbox_label] = font.render(
            checkbox_label, True, (255, 255, 255)
        )
        labal_position[1] += font_size[1] * 1.7
    for button_label in buttons.keys():
        widgets["button_boxes"][button_label] = pg.Rect(
            labal_position[0],
            labal_position[1],
            font.size(button_label)[0] * 1.2,
            font.size(button_label)[1] * 1.5,
        )
        widgets["button_label"][button_label] = font.render(
            button_label, True, (255, 255, 255)
        )
        labal_position[1] += font_size[1] * 1.7
    widgets["active_box"] = widgets["input_boxes"][
        0
    ]  # Set the cursor in the first input box
    widgets["state"] = State.CONFIGURING
    widgets["pins"] = calculate_pins(
        widgets["image_square_size"],
        params["Radius in Pixels"],
        params["Number of Pins"],
    )


def create_information_widgets(widgets):
    buttons = {
        "Back": create_config_widgets,
        "Stop": stop_drawing,
        "Pause": pause_drawing,
        "Resume": resume_drawing,
    }
    widgets["input_boxes"] = []
    widgets["checkboxes"] = {}
    widgets["buttons"] = buttons
    widgets["button_boxes"] = {}
    widgets["button_label"] = {}


def stop_drawing(widgets):
    """
    Stop the drawing process and start the server.

    Args:
        widgets (dict): A dictionary containing all the widgets and parameters.
    """
    data_list = []
    widgets["state"] = State.SERVING
    if widgets["use_pin_index"]:
        data_list = widgets["steps_index"]
    else:
        data_list = to_real_coordinates(
                widgets["steps"],
                widgets["image_square_size"],
                widgets["parameters"]["Radius in Pixels"],
                widgets["parameters"]["Radius in milimeter"],
            )
    widgets["server"] = Server(
        data_list,
        "0.0.0.0",
        65432,
    )

def pause_drawing(widgets):
    widgets["state"] = State.PAUSING
    widgets["buttons"]["Resume"] = resume_drawing


def resume_drawing(widgets):
    widgets["state"] = State.DRAWING
    widgets["buttons"].pop("Resume", None)


def move_to_next_box(widgets):
    if widgets["active_box"] in widgets["input_boxes"]:
        current_index = widgets["input_boxes"].index(widgets["active_box"])
        next_index = (current_index + 1) % (len(widgets["input_boxes"]) + 1)
        if next_index < len(widgets["input_boxes"]):
            widgets["active_box"] = widgets["input_boxes"][next_index]
        else:
            widgets["active_box"] = None  # Move to submit button
    elif widgets["active_box"] is None:
        widgets["active_box"] = widgets["input_boxes"][
            0
        ]  # Move back to the first input box


def get_param_name(widgets):
    index = widgets["input_boxes"].index(widgets["active_box"])
    return list(widgets["parameters"].keys())[index]


def submit_parameters(widgets):
    # Convert parameters to integers
    if widgets["image"] is None:
        return
    parameters = widgets["parameters"]
    for key in parameters:
        if parameters[key] == "":
            parameters[key] = "0"  # Default to 0 if empty
        parameters[key] = int(parameters[key])
    if widgets["image"].mode != "L":
        widgets["image"] = widgets["image"].convert("L")
    if widgets["init_array"] is None:
        widgets["init_array"] = np.array(widgets["image"])
    widgets["state"] = State.DRAWING
    create_information_widgets(widgets)
    init_drawing(widgets)
    widgets["steps"] = []
    widgets["steps_index"] = []
    widgets["steps"].append(widgets["pins"][widgets["current_index"]])
    widgets["steps_index"].append(widgets["current_index"])
    widgets["buttons"].pop("Resume", None)


def select_image(widgets):
    image = None
    # Use tkinter to open a file dialog
    screen_size = widgets["image_square_size"]
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    if file_path:
        # image = Image.open(file_path).convert('L')
        image = Image.open(file_path)
        img_width, img_height = image.size
        # Scale the image if it is smaller than the square
        if img_width < screen_size or img_height < screen_size:
            scale_factor = max(screen_size / img_width, screen_size / img_height)
            new_size = (int(img_width * scale_factor), int(img_height * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
            img_width, img_height = new_size

        if img_width > screen_size or img_height > screen_size:
            # Calculate the cropping box
            left = (img_width - screen_size) // 2
            top = (img_height - screen_size) // 2
            right = left + screen_size
            bottom = top + screen_size
            image = image.crop((left, top, right, bottom))
        widgets["image"] = image
        widgets["image_original"] = image.copy()
        process_image(widgets)

    root.destroy()
    return image


def apply_circle_mask(image, center, radius):
    # Convert to RGBA if image is not already in RGBA mode
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create a mask with the same size as the original image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw white circle on black background
    draw.ellipse(
        [
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
        ],
        fill=255,
    )

    # Set alpha channel using the mask
    image.putalpha(mask)

    # Convert back to RGB on white background
    white_background = Image.new("RGB", image.size, (255, 255, 255))
    white_background.paste(image, mask=image.split()[3])  # Use alpha channel as mask

    return white_background


def process_image(widgets):
    edge_params = {
        "canny_low": 100,
        "canny_high": 160,
        "adaptive_block": 9,
        "adaptive_c": 2,
        "blur_kernel": (5, 5),
        "dilate_kernel": np.ones((2, 2), np.uint8),
        "erode_kernel": np.ones((1, 1), np.uint8),
    }

    if widgets["image"] is not None:
        widgets["image"] = widgets["image_original"].copy()
        # Convert to grayscale
        if widgets["checkboxes"]["Remove Background"]:
            # Remove background
            widgets["image"] = remove(widgets["image"])
            # Convert background to white
            bg = Image.new("RGBA", widgets["image"].size, (255, 255, 255, 255))
            widgets["image"] = Image.alpha_composite(bg, widgets["image"])

        widgets["image"] = widgets["image"].convert("L")
        image_array = np.array(widgets["image"])
        widgets["init_array"] = image_array.copy()
        if widgets["checkboxes"]["denoise"]:
            # image_array = cv2.GaussianBlur(image_array, edge_params['blur_kernel'], 0)
            image_array = cv2.bilateralFilter(image_array, 9, 75, 75)

        if widgets["checkboxes"]["Canny edge detection"]:
            # Method 1: Canny edge detection
            image_array = cv2.Canny(
                image_array, edge_params["canny_low"], edge_params["canny_high"]
            )

        if widgets["checkboxes"]["Adaptive thresholding"]:
            # Method 2: Adaptive thresholding
            image_array = cv2.adaptiveThreshold(
                image_array,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                edge_params["adaptive_block"],
                edge_params["adaptive_c"],
            )
        if widgets["checkboxes"]["Global Thresholding"]:
            _, image_array = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)

        if widgets["checkboxes"]["Line Thickness"]:
            # Adjust line thickness if needed
            image_array = cv2.dilate(
                image_array, edge_params["dilate_kernel"], iterations=3 - 1
            )
        if image_array[0][0] == 0:
            # Invert colors to get black lines on white background
            image_array = cv2.bitwise_not(image_array)

        widgets["image"] = apply_circle_mask(
            Image.fromarray(image_array),
            (widgets["image_square_size"] // 2, widgets["image_square_size"] // 2),
            widgets["parameters"]["Radius in Pixels"],
        )
        widgets["processed_array"] = np.array(widgets["image"].convert("L"))
        widgets["init_array"] = np.array(
            apply_circle_mask(
                Image.fromarray(widgets["init_array"]),
                (widgets["image_square_size"] // 2, widgets["image_square_size"] // 2),
                widgets["parameters"]["Radius in Pixels"],
            ).convert("L")
        )
        if widgets["checkboxes"]["Use Processed Image Only"]:
            widgets["init_array"] = widgets["processed_array"].copy()
            widgets["processed_array"] = None
        widgets["use_pin_index"] = widgets["checkboxes"]["Use Pin Index"]


def calculate_pins(squareSize, radius, num_pins):
    pins = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        x = squareSize // 2 + radius * math.cos(angle)
        y = squareSize // 2 + radius * math.sin(angle)
        pins.append((x, y))
    return pins


def to_real_coordinates(pins, squareSize, radius_pixel, radius_milimeter):
    """
    Convert pins coordinates relate to application window with central point at (squareSize // 2,squareSize // 2) and radius in pixels
     to real world coordinates with central point at (0,0) and radius in milimeter.
    """
    real_pins = []
    for pin in pins:
        x = radius_milimeter * (pin[0] - squareSize // 2) / radius_pixel
        y = radius_milimeter * (pin[1] - squareSize // 2) / radius_pixel
        real_pins.append((x, y))
    return real_pins


def calculate_line_darkness(img_array, x1, y1, x2, y2):
    """
    Calculate the darkness of pixel values along a line between two points in a grayscale image.

    Args:
        image: PIL Image in 'L' mode (grayscale)
        x1, y1: Starting point coordinates
        x2, y2: Ending point coordinates

    Returns:
        tuple: (darkness of pixel values, list of coordinates along the line)
    """
    pixels = []
    pixel_sum = 0

    # Calculate changes and steps
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Determine direction
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1

    # Current positions
    x = x1
    y = y1

    # Determine whether to drive the algorithm by x or y
    if dx > dy:
        # Drive by x
        error = dx / 2
        while x != x2 + x_step:
            if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
                pixel_sum += img_array[y, x]
                pixels.append((x, y))

            error -= dy
            if error < 0:
                y += y_step
                error += dx
            x += x_step
    else:
        # Drive by y
        error = dy / 2
        while y != y2 + y_step:
            if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
                pixel_sum += img_array[y, x]
                pixels.append((x, y))

            error -= dx
            if error < 0:
                x += x_step
                error += dy
            y += y_step

    if pixel_sum == 0:
        return 255, []
    return pixel_sum / len(pixels), pixels


def find_best_next_pin(widgets, current_index, last_index):
    best_darkness = 255
    best_line = None
    best_pin = None
    current_pin = widgets["pins"][current_index]
    steps = copy.copy(widgets["steps"])
    target_pins = copy.copy(widgets["pins"])
    if last_index is not None:
        if last_index > current_index:
            target_pins.pop(last_index)
            target_pins.pop(current_index)
        else:
            target_pins.pop(current_index)
            target_pins.pop(last_index)
    else:
        target_pins.pop(current_index)

    # remove used steps
    while True:
        try:
            index = steps.index(current_pin)
            if index > 0:
                target_pins.remove(steps[index - 1])
            if index < len(steps) - 1:
                target_pins.remove(steps[index + 1])
            steps.pop(index)
        except ValueError:
            break

    # get 30 random pins from target_pins and save to random_pins
    random_pins = random.sample(target_pins, min(30, len(target_pins)))
    # for pin in target_pins:
    for pin in random_pins:
        darkness, line = calculate_line_darkness(
            widgets["init_array"],
            int(current_pin[0]),
            int(current_pin[1]),
            int(pin[0]),
            int(pin[1]),
        )
        if (
            darkness < best_darkness
            and len(line) > widgets["parameters"]["Shortest Line in Pixels"]
            and steps.count(pin) < widgets["parameters"]["Max Pin Usage"]
        ):
            best_darkness = darkness
            best_line = line
            best_pin = pin

    return best_pin, best_line


def update_image_array(widgets, line):
    for x, y in line:
        if widgets["init_array"][y, x] != 255:
            if widgets["processed_array"] is not None:
                if (
                    widgets["processed_array"][y, x] == 0
                    and widgets["init_array"][y, x] < 200
                ):
                    widgets["init_array"][y, x] = (
                        widgets["init_array"][y, x]
                        + (255 - widgets["init_array"][y, x]) // 2
                    )
                else:
                    widgets["init_array"][y, x] = 255
            elif widgets["init_array"][y, x] < 200:
                # widgets['init_array'][y, x] = min(int(widgets['init_array'][y, x] * widgets['decay']), 255)
                widgets["init_array"][y, x] = min(
                    int(widgets["init_array"][y, x] + 100), 255
                )
            else:
                widgets["init_array"][y, x] = 255


def draw_line(screen, line, color=(0, 0, 0)):
    for x, y in line:
        screen.set_at((x, y), color)


def draw_config_widgets(widgets):
    surface = widgets["information_surface"]
    font = widgets["font"]
    parameters = widgets["parameters"]
    # Clear the surface
    surface.fill((30, 30, 30))

    for label, pos in widgets["labels"]:
        surface.blit(label, pos)

    for i, box in enumerate(widgets["input_boxes"]):
        # Clear the input box area
        pg.draw.rect(surface, (30, 30, 30), box)

        # Draw the input box border
        pg.draw.rect(surface, (255, 255, 255), box, 2)

        param_name = list(parameters.keys())[i]
        txt_surface = font.render(str(parameters[param_name]), True, (255, 255, 255))
        surface.blit(txt_surface, (box.x + 5, box.y + 5))

        # Draw cursor if this is the active box
        if box == widgets.get("active_box") and widgets.get("cursor_visible"):
            cursor_x = box.x + 5 + txt_surface.get_width()
            cursor_y = box.y + 5
            pg.draw.line(
                surface,
                (255, 255, 255),
                (cursor_x, cursor_y),
                (cursor_x, cursor_y + txt_surface.get_height()),
                2,
            )

    for checkbox_label in widgets["checkboxes"]:
        checkbox_box = widgets["checkbox_boxes"][checkbox_label]
        pg.draw.rect(surface, (255, 255, 255), checkbox_box, 2)
        if widgets["checkboxes"][checkbox_label]:
            pg.draw.line(
                surface,
                (255, 255, 255),
                (checkbox_box.left, checkbox_box.top),
                (checkbox_box.right, checkbox_box.bottom),
                2,
            )
            pg.draw.line(
                surface,
                (255, 255, 255),
                (checkbox_box.left, checkbox_box.bottom),
                (checkbox_box.right, checkbox_box.top),
                2,
            )
        surface.blit(
            widgets["checkbox_labels"][checkbox_label],
            (checkbox_box.right + 5, checkbox_box.top),
        )

    for button_label in widgets["buttons"]:
        surface.blit(
            widgets["button_label"][button_label],
            (
                widgets["button_boxes"][button_label].x + 5,
                widgets["button_boxes"][button_label].y + 5,
            ),
        )
        pg.draw.rect(surface, (255, 255, 255), widgets["button_boxes"][button_label], 2)


def draw_preview_image(widgets):
    surface = widgets["string_drawing_surface"]
    image = widgets["image"]
    parameters = widgets["parameters"]
    surface.fill(widgets["color_white"])
    # Draw the selected image if available
    if image:
        image_surface = pg.image.fromstring(
            image.convert("RGB").tobytes(), image.size, "RGB"
        )
        surface.blit(image_surface, (0, 0))

    square_size = surface.get_width()
    center = (square_size // 2, square_size // 2)
    radius = parameters["Radius in Pixels"]

    # Draw circle outline
    pg.draw.circle(surface, (200, 200, 200), center, radius, 1)

    # Draw pins
    for pin in widgets["pins"]:
        pg.draw.circle(surface, (100, 100, 100), pin, 2)


def init_drawing(widgets):
    surface = widgets["string_drawing_surface"]
    parameters = widgets["parameters"]
    radius = parameters["Radius in Pixels"]
    square_size = widgets["image_square_size"]
    pins = widgets["pins"]
    center = (square_size // 2, square_size // 2)

    surface.fill(widgets["color_white"])
    # Draw circle outline
    pg.draw.circle(surface, (200, 200, 200), center, radius, 1)

    # Draw pins
    for pin in pins:
        pg.draw.circle(surface, (100, 100, 100), pin, 2)

    pg.display.flip()


def draw_string(widgets):
    surface = widgets["string_drawing_surface"]
    string_color = widgets["color_black"]
    next_pin, line = find_best_next_pin(
        widgets, widgets["current_index"], widgets["last_index"]
    )
    if next_pin is not None:
        widgets["last_index"] = widgets["current_index"]
        widgets["current_index"] = widgets["pins"].index(next_pin)
        update_image_array(widgets, line)
        widgets["total_length_in_pixels"] += len(line)
        widgets["steps"].append(next_pin)
        widgets["steps_index"].append(widgets["current_index"])
        draw_line(surface, line, string_color)
    return next_pin


def draw_information(widgets):
    surface = widgets["information_surface"]
    font = widgets["font"]
    color = widgets["color_white"]
    margin = widgets["image_square_size"] * 0.05
    position = [margin, margin]
    fontsize = font.size("1000000")
    # Clear the surface
    surface.fill((30, 30, 30))
    surface.blit(
        font.render(
            "Diameter = "
            + str(2 * widgets["parameters"]["Radius in milimeter"] / 1000)
            + " meters",
            True,
            color,
        ),
        position,
    )
    position[1] += fontsize[1] * 2
    surface.blit(
        font.render(
            "Nails = " + str(widgets["parameters"]["Number of Pins"]), True, color
        ),
        position,
    )
    position[1] += fontsize[1] * 2
    surface.blit(
        font.render("Number of Lines = " + str(len(widgets["steps"]) - 1), True, color),
        position,
    )
    position[1] += fontsize[1] * 2
    surface.blit(
        font.render(
            "Total length = "
            + str(
                widgets["total_length_in_pixels"]
                / widgets["parameters"]["Radius in Pixels"]
                * widgets["parameters"]["Radius in milimeter"]
                / 1000
            )
            + " meters",
            True,
            color,
        ),
        position,
    )
    position[1] += fontsize[1] * 2
    for button_label in widgets["buttons"].keys():
        widgets["button_boxes"][button_label] = pg.Rect(
            position[0],
            position[1],
            font.size(button_label)[0] * 1.2,
            font.size(button_label)[1] * 1.5,
        )
        widgets["button_label"][button_label] = font.render(button_label, True, color)
        font_size = font.size(button_label)
        surface.blit(
            widgets["button_label"][button_label],
            (
                widgets["button_boxes"][button_label].x + 5,
                widgets["button_boxes"][button_label].y + 5,
            ),
        )
        pg.draw.rect(surface, color, widgets["button_boxes"][button_label], 2)
        position[1] += font_size[1] * 2
    new_size = (
        int(widgets["image_square_size"] // 5),
        int(widgets["image_square_size"] // 5),
    )
    image_position = [
        surface.get_width() - new_size[0],
        surface.get_height() - new_size[1],
    ]
    if widgets["processed_array"] is not None:
        image = Image.fromarray(widgets["processed_array"])
        image = image.resize(new_size, Image.LANCZOS)
        image_surface = pg.image.fromstring(
            image.convert("RGB").tobytes(), image.size, "RGB"
        )
        surface.blit(image_surface, image_position)
        image_position[0] -= new_size[0]
    if widgets["init_array"] is not None:
        image = Image.fromarray(widgets["init_array"])
        image = image.resize(new_size, Image.LANCZOS)
        image_surface = pg.image.fromstring(
            image.convert("RGB").tobytes(), image.size, "RGB"
        )
        surface.blit(image_surface, image_position)


def draw_serving(widgets):
    surface = widgets["information_surface"]
    font = widgets["font"]
    color = widgets["color_white"]
    margin = widgets["image_square_size"] * 0.05
    position = [margin, margin]
    fontsize = font.size("1000000")
    server = widgets["server"]

    # Clear the surface
    surface.fill((30, 30, 30))
    if server is None:
        surface.blit(font.render("Server is not running", True, color), position)
    else:
        surface.blit(font.render(server.get_state(), True, color), position)
        position[1] += fontsize[1] * 2
        surface.blit(
            font.render(
                "Total Number of Lines = " + str(len(widgets["steps"]) - 1), True, color
            ),
            position,
        )
        position[1] += fontsize[1] * 2
        surface.blit(
            font.render(
                "Coordinates left to be sent = " + str(server.get_data_length()),
                True,
                color,
            ),
            position,
        )
        position[1] += fontsize[1] * 2


def handle_event(widgets, event):
    square_size = widgets["image_square_size"]
    if event.type == pg.MOUSEBUTTONDOWN:
        # Iterate over a copy of the keys to avoid modifying the dictionary during iteration
        for button_label in list(widgets["buttons"].keys()):
            if widgets["button_boxes"][button_label].collidepoint(
                (event.pos[0] - square_size, event.pos[1])
            ):
                widgets["buttons"][button_label](widgets)
        for box in widgets["input_boxes"]:
            if box.collidepoint((event.pos[0] - square_size, event.pos[1])):
                widgets["active_box"] = box
                break
        for checkbox_label in list(widgets["checkboxes"].keys()):
            if widgets["checkbox_boxes"][checkbox_label].collidepoint(
                (event.pos[0] - square_size, event.pos[1])
            ):
                widgets["checkboxes"][checkbox_label] = not widgets["checkboxes"][
                    checkbox_label
                ]
                process_image(widgets)
                break
    elif event.type == pg.KEYDOWN:
        if widgets.get("active_box"):
            param_name = get_param_name(widgets)
            if event.key == pg.K_RETURN:
                widgets["active_box"] = None
            elif event.key == pg.K_BACKSPACE:
                current_value = str(widgets["parameters"][param_name])
                widgets["parameters"][param_name] = (
                    int(current_value[:-1]) if current_value[:-1] else 0
                )
                if param_name == "Number of Pins" or param_name == "Radius in Pixels":
                    widgets["pins"] = calculate_pins(
                        widgets["image_square_size"],
                        widgets["parameters"]["Radius in Pixels"],
                        widgets["parameters"]["Number of Pins"],
                    )
                    process_image(widgets)
            elif event.key == pg.K_TAB:
                move_to_next_box(widgets)
            elif event.unicode.isdigit():
                current_value = str(widgets["parameters"][param_name])
                widgets["parameters"][param_name] = int(current_value + event.unicode)
                if param_name == "Number of Pins" or param_name == "Radius in Pixels":
                    widgets["pins"] = calculate_pins(
                        widgets["image_square_size"],
                        widgets["parameters"]["Radius in Pixels"],
                        widgets["parameters"]["Number of Pins"],
                    )
                    process_image(widgets)


def main():
    pg.init()
    widgets = init_widgets()
    screen = widgets["window"]
    square_size = widgets["image_square_size"]
    create_config_widgets(widgets)

    while widgets["state"] != State.QUITING:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                widgets["state"] = State.QUITING
            handle_event(widgets, event)
        if widgets["state"] == State.SERVING and widgets["server"] is not None:
            widgets["server"].handle_events()
        match widgets["state"]:
            case State.CONFIGURING:
                draw_config_widgets(widgets)
                screen.blit(widgets["information_surface"], (square_size, 0))
                draw_preview_image(widgets)
                screen.blit(widgets["string_drawing_surface"], (0, 0))
                pg.display.flip()
            case State.DRAWING:
                next_pin = draw_string(widgets)
                if next_pin is None:
                    stop_drawing(widgets)
                max_lines = widgets["parameters"]["Number of Lines"]
                if (len(widgets["steps"]) - 1) % max_lines == 0:
                    pause_drawing(widgets)
                screen.blit(widgets["string_drawing_surface"], (0, 0))
                draw_information(widgets)
                screen.blit(
                    widgets["information_surface"], (widgets["image_square_size"], 0)
                )
                pg.display.flip()
            case State.PAUSING:
                screen.blit(widgets["string_drawing_surface"], (0, 0))
                draw_information(widgets)
                screen.blit(
                    widgets["information_surface"], (widgets["image_square_size"], 0)
                )
                pg.display.flip()
            case State.SERVING:
                draw_serving(widgets)
                screen.blit(
                    widgets["information_surface"], (widgets["image_square_size"], 0)
                )
                pg.display.flip()
            case State.QUITING:
                if widgets["server"] is not None:
                    widgets["server"].stop()

    if widgets["image"]:
        print(len(list(widgets["image"].getdata())))


if __name__ == "__main__":
    main()
