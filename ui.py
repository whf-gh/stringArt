import pygame as pg
import copy
import numpy as np
from tkinter import Tk, filedialog # Keep for select_image_callback's potential direct use if not fully abstracted
from PIL import Image, ImageDraw # Keep for draw_preview_image if it uses PIL directly, and apply_circle_mask_callback
# from rembg import remove # No longer needed here
# import cv2 # No longer needed here

# Assuming State enum will be imported from stringart.py or a common module
# For now, if you have a State class/enum, ensure it's accessible here.
# from stringart import State # Placeholder

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
    # widgets["state"] = State.CONFIGURING # State will be managed by main module
    widgets["image"] = None # This will be a pygame surface or None
    widgets["image_original"] = None # This might store the path or a PIL image if needed for re-processing
    # Non-UI related initializations moved to main stringart.py
    # widgets["decay"] = 1.5
    # widgets["init_array"] = None
    # widgets["processed_array"] = None
    # widgets["current_index"] = 0
    # widgets["last_index"] = None
    # widgets["pins"] = []
    # widgets["use_pin_index"] = True
    # widgets["steps"] = []
    # widgets["steps_index"] = []
    # widgets["lines"] = []
    # widgets["total_length_in_pixels"] = 0
    # widgets["server"] = None
    return widgets


def create_config_widgets(widgets, select_image_callback, submit_parameters_callback, initial_params, initial_checkboxes, calculate_pins_callback, process_image_callback):
    params = initial_params
    buttons = {"Select Image": select_image_callback, "Submit": submit_parameters_callback}
    checkboxes = initial_checkboxes
    
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
    
    widgets["labels"] = [] # Clear previous labels
    widgets["input_boxes"] = [] # Clear previous input boxes
    widgets["button_boxes"] = {} # Clear previous button boxes
    widgets["button_label"] = {} # Clear previous button labels
    widgets["checkbox_boxes"] = {} # Clear previous checkbox boxes
    widgets["checkbox_labels"] = {} # Clear previous checkbox labels

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
    if widgets["input_boxes"]: # Check if input_boxes is not empty
        widgets["active_box"] = widgets["input_boxes"][0]
    else:
        widgets["active_box"] = None
    # widgets["state"] = State.CONFIGURING # State will be managed by main module
    # The 'pins' calculation is a core logic, not directly UI.
    # The UI might display pins, but their calculation is tied to core parameters.
    # stringart.py will call calculate_pins and store it in widgets["pins"]
    # create_config_widgets should then use widgets["pins"] if available for display purposes or
    # rely on stringart.py to update it.
    # For now, let's assume stringart.py handles pin calculation and updates widgets["pins"].
    # If calculate_pins_callback was meant to trigger a re-calculation and update,
    # that logic will be in stringart.py.
    # widgets["pins"] = calculate_pins_callback( 
    #     widgets["image_square_size"],
    #     params["Radius in Pixels"],
    #     params["Number of Pins"],
    # )
    # Instead, ensure 'pins' are available in widgets for drawing, populated by stringart.py
    if "pins" not in widgets: # Initialize if not present, stringart.py should fill this
        widgets["pins"] = []
    # If params change, stringart.py should detect this (e.g. in submit_parameters or handle_event if param is edited)
    # and call calculate_pins, then update widgets["pins"], then tell UI to redraw.

    # The process_image_callback is used in handle_event when checkboxes change.
    # The select_image_callback is used for the "Select Image" button.
    # These callbacks will be connected by stringart.py to its own orchestrator functions
    # which in turn will call the respective functions from image_processing.py.
    pass # No direct change here, but noting the responsibility of stringart.py


def create_information_widgets(widgets, back_callback, stop_callback, pause_callback, resume_callback):
        params["Radius in Pixels"],
        params["Number of Pins"],
    )


def create_information_widgets(widgets, back_callback, stop_callback, pause_callback, resume_callback):
    buttons = {
        "Back": back_callback,
        "Stop": stop_callback,
        "Pause": pause_callback,
        "Resume": resume_callback,
    }
    widgets["input_boxes"] = []
    widgets["checkboxes"] = {}
    widgets["buttons"] = buttons
    widgets["button_boxes"] = {}
    widgets["button_label"] = {}

def create_serving_widgets(widgets, back_callback, reset_callback):
    buttons = {
        "Back": back_callback,
        "Reset": reset_callback,
    }
    widgets["input_boxes"] = []
    widgets["checkboxes"] = {}
    widgets["buttons"] = buttons
    widgets["button_boxes"] = {}
    widgets["button_label"] = {}


def move_to_next_box(widgets):
    if not widgets["input_boxes"]: # Check if input_boxes is empty
        return
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
    if not widgets["active_box"] or not widgets["input_boxes"]: # Check if active_box or input_boxes is None/empty
        return None
    try:
        index = widgets["input_boxes"].index(widgets["active_box"])
        return list(widgets["parameters"].keys())[index]
    except ValueError: # active_box might not be in input_boxes if it was cleared
        return None

# select_image, apply_circle_mask, and process_image functions have been moved to image_processing.py
# ui.py will rely on callbacks provided during widget creation to trigger these operations.
# stringart.py will be responsible for implementing these callbacks and calling the
# functions from image_processing.py, then updating the widgets dictionary,
# and finally telling the UI to redraw if necessary.

def draw_config_widgets(widgets):
    surface = widgets["information_surface"]
    font = widgets["font"]
    parameters = widgets["parameters"]
    surface.fill((30, 30, 30))

    for label, pos in widgets["labels"]:
        surface.blit(label, pos)

    for i, box in enumerate(widgets["input_boxes"]):
        pg.draw.rect(surface, (30, 30, 30), box)
        pg.draw.rect(surface, (255, 255, 255), box, 2)
        param_name = list(parameters.keys())[i]
        txt_surface = font.render(str(parameters[param_name]), True, (255, 255, 255))
        surface.blit(txt_surface, (box.x + 5, box.y + 5))
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
    # widgets["image"] should be a Pygame surface, or None.
    # It's populated by stringart.py after image selection/processing.
    pg_image_surface = widgets.get("image") 
    parameters = widgets["parameters"]
    surface.fill(widgets["color_white"])

    if pg_image_surface:
        # Ensure pg_image_surface is a pygame.Surface
        # If it was stored as PIL Image in widgets["image"], it needs conversion before blitting.
        # However, the expectation is that stringart.py prepares a pygame surface for the UI.
        surface.blit(pg_image_surface, (0, 0))

    square_size = surface.get_width()
    center = (square_size // 2, square_size // 2)
    # Ensure "Radius in Pixels" is present and valid
    radius = parameters.get("Radius in Pixels", square_size // 4) # Default if not found

    pg.draw.circle(surface, (200, 200, 200), center, radius, 1)
    if "pins" in widgets and widgets["pins"]: # Check if pins exist
        for pin in widgets["pins"]:
            pg.draw.circle(surface, (100, 100, 100), pin, 2)


def init_drawing(widgets): # Initializes the drawing surface
    surface = widgets["string_drawing_surface"]
    parameters = widgets["parameters"]
    # Ensure "Radius in Pixels" is present and valid
    radius = parameters.get("Radius in Pixels", widgets["image_square_size"] // 4)
    square_size = widgets["image_square_size"]
    # Ensure "pins" are calculated and present if needed by other logic
    pins = widgets.get("pins", []) # Use .get for safety
    center = (square_size // 2, square_size // 2)

    surface.fill(widgets["color_white"])
    pg.draw.circle(surface, (200, 200, 200), center, radius, 1)
    for pin in pins:
        pg.draw.circle(surface, (100, 100, 100), pin, 2)
    pg.display.flip() # This might be better called in the main loop


def draw_information(widgets):
    surface = widgets["information_surface"]
    font = widgets["font"]
    color = widgets["color_white"]
    margin = widgets["image_square_size"] * 0.05
    position = [margin, margin]
    fontsize = font.size("1000000")
    surface.fill((30, 30, 30))
    surface.blit(
        font.render(
            "Diameter = "
            + str(2 * widgets["parameters"].get("Radius in milimeter", 0) / 1000) # Use .get
            + " meters",
            True,
            color,
        ),
        position,
    )
    position[1] += fontsize[1] * 2
    surface.blit(
        font.render(
            "Nails = " + str(widgets["parameters"].get("Number of Pins", 0)), True, color # Use .get
        ),
        position,
    )
    position[1] += fontsize[1] * 2
    # 'steps' might not be in widgets if it's purely UI related now
    surface.blit(
        font.render("Number of Lines = " + str(len(widgets.get("steps", [])) - 1), True, color), # Use .get
        position,
    )
    position[1] += fontsize[1] * 2
    # 'total_length_in_pixels' might not be in widgets
    surface.blit(
        font.render(
            "Total length = "
            + str(
                widgets.get("total_length_in_pixels", 0) # Use .get
                / widgets["parameters"].get("Radius in Pixels", 1) # Use .get, avoid division by zero
                * widgets["parameters"].get("Radius in milimeter", 0) # Use .get
                / 1000
            )
            + " meters",
            True,
            color,
        ),
        position,
    )
    position[1] += fontsize[1] * 2
    
    widgets["button_boxes"] = {} # Clear previous button boxes
    widgets["button_label"] = {} # Clear previous button labels

    for button_label in widgets["buttons"].keys():
        widgets["button_boxes"][button_label] = pg.Rect(
            position[0],
            position[1],
            font.size(button_label)[0] * 1.2,
            font.size(button_label)[1] * 1.5,
        )
        widgets["button_label"][button_label] = font.render(button_label, True, color)
        font_size_btn = font.size(button_label) # Renamed to avoid conflict
        surface.blit(
            widgets["button_label"][button_label],
            (
                widgets["button_boxes"][button_label].x + 5,
                widgets["button_boxes"][button_label].y + 5,
            ),
        )
        pg.draw.rect(surface, color, widgets["button_boxes"][button_label], 2)
        position[1] += font_size_btn[1] * 2 # Use renamed variable

    new_size = (
        int(widgets["image_square_size"] // 5),
        int(widgets["image_square_size"] // 5),
    )
    image_position = [
        surface.get_width() - new_size[0],
        surface.get_height() - new_size[1],
    ]
    # 'processed_array' and 'init_array' are numpy arrays.
    # For display, they need to be converted to PIL Images then to Pygame surfaces.
    # This should ideally be handled by stringart.py, which then stores
    # the pygame surfaces in widgets (e.g., widgets["display_processed_surface"], widgets["display_init_surface"])
    
    # Assuming stringart.py prepares these as pygame surfaces:
    display_processed_surface = widgets.get("display_processed_surface")
    display_init_surface = widgets.get("display_init_surface")

    if display_processed_surface:
        surface.blit(display_processed_surface, image_position)
        image_position[0] -= new_size[0] # Adjust for next image

    if display_init_surface:
        surface.blit(display_init_surface, image_position)


def draw_serving(widgets):
    surface = widgets["information_surface"]
    font = widgets["font"]
    color = widgets["color_white"]
    margin = widgets["image_square_size"] * 0.05
    position = [margin, margin]
    fontsize = font.size("1000000")
    server = widgets.get("server") # Use .get for server

    surface.fill((30, 30, 30))
    if server is None:
        surface.blit(font.render("Server is not running", True, color), position)
    else:
        surface.blit(font.render(server.get_state(), True, color), position)
        position[1] += fontsize[1] * 2
        surface.blit(
            font.render(
                "Total Number of Lines = " + str(len(widgets.get("steps", [])) - 1), True, color # Use .get
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

    widgets["button_boxes"] = {} # Clear previous button boxes
    widgets["button_label"] = {} # Clear previous button labels
    # Add buttons for serving screen if any, similar to draw_information
    for button_label in widgets.get("buttons", {}).keys(): # Use .get for buttons
        # This part might need adjustment based on how buttons are defined for serving screen
        # Assuming button drawing logic is similar to draw_information
        widgets["button_boxes"][button_label] = pg.Rect(
            position[0],
            position[1],
            font.size(button_label)[0] * 1.2,
            font.size(button_label)[1] * 1.5,
        )
        widgets["button_label"][button_label] = font.render(button_label, True, color)
        font_size_btn = font.size(button_label)
        surface.blit(
            widgets["button_label"][button_label],
            (
                widgets["button_boxes"][button_label].x + 5,
                widgets["button_boxes"][button_label].y + 5,
            ),
        )
        pg.draw.rect(surface, color, widgets["button_boxes"][button_label], 2)
        position[1] += font_size_btn[1] * 2


def handle_event(widgets, event, process_image_callback, calculate_pins_callback, main_event_handler_callback):
    square_size = widgets["image_square_size"]
    if event.type == pg.MOUSEBUTTONDOWN:
        # Adjust mouse position for information_surface
        original_pos = event.pos
        adjusted_pos = (event.pos[0] - square_size, event.pos[1]) if event.pos[0] >= square_size else event.pos

        for button_label in list(widgets.get("buttons", {}).keys()): # Use .get
            # Check if button_boxes for this button_label exists
            if button_label in widgets["button_boxes"]:
                if widgets["button_boxes"][button_label].collidepoint(adjusted_pos):
                    # Pass the main widgets dict to the callback
                    widgets["buttons"][button_label](widgets) 
                    return # Event handled

        for box in widgets.get("input_boxes", []): # Use .get
            if box.collidepoint(adjusted_pos):
                widgets["active_box"] = box
                return # Event handled
        
        for checkbox_label in list(widgets.get("checkboxes", {}).keys()): # Use .get
             # Check if checkbox_boxes for this checkbox_label exists
            if checkbox_label in widgets["checkbox_boxes"]:
                if widgets["checkbox_boxes"][checkbox_label].collidepoint(adjusted_pos):
                    widgets["checkboxes"][checkbox_label] = not widgets["checkboxes"][checkbox_label]
                    # Call process_image via callback from main module
                    process_image_callback(widgets) 
                    return # Event handled

    elif event.type == pg.KEYDOWN:
        active_box_param_name = get_param_name(widgets) # Renamed for clarity
        if active_box_param_name: # Check if a parameter name was returned
            if event.key == pg.K_RETURN:
                widgets["active_box"] = None
            elif event.key == pg.K_BACKSPACE:
                current_value = str(widgets["parameters"][active_box_param_name])
                widgets["parameters"][active_box_param_name] = (
                    int(current_value[:-1]) if current_value[:-1] else 0
                )
                if active_box_param_name == "Number of Pins" or active_box_param_name == "Radius in Pixels":
                    # Call calculate_pins and process_image via callbacks
                    widgets["pins"] = calculate_pins_callback(
                        widgets["image_square_size"],
                        widgets["parameters"]["Radius in Pixels"],
                        widgets["parameters"]["Number of Pins"],
                    )
                    process_image_callback(widgets)
            elif event.key == pg.K_TAB:
                move_to_next_box(widgets)
            elif event.unicode.isdigit():
                current_value = str(widgets["parameters"][active_box_param_name])
                widgets["parameters"][active_box_param_name] = int(current_value + event.unicode)
                if active_box_param_name == "Number of Pins" or active_box_param_name == "Radius in Pixels":
                    # Call calculate_pins and process_image via callbacks
                    widgets["pins"] = calculate_pins_callback(
                        widgets["image_square_size"],
                        widgets["parameters"]["Radius in Pixels"],
                        widgets["parameters"]["Number of Pins"],
                    )
                    process_image_callback(widgets)
            return # Event handled by UI input fields
    
    # If event was not handled by UI elements, pass to main event handler
    main_event_handler_callback(widgets, event)
