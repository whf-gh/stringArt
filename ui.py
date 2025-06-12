import pygame as pg
import copy
import numpy as np
# from tkinter import Tk, filedialog # Removed as select_image moved to image_processing.py
from PIL import Image, ImageDraw # Keep for draw_preview_image if it uses PIL directly, and apply_circle_mask_callback
# from rembg import remove # No longer needed here
# import cv2 # No longer needed here

# Assuming State enum will be imported from stringart.py or a common module
# For now, if you have a State class/enum, ensure it's accessible here.
# from stringart import State # Placeholder

# Define scrollbar configurations at module level for broader access
SCROLLBAR_PARAM_CONFIGS = {
    'Canny Low': {'min': 0, 'max': 255},
    'Canny High': {'min': 0, 'max': 255},
    'Adaptive Block': {'min': 3, 'max': 51}, # Should be odd, handle in update logic
    'Adaptive C': {'min': 0, 'max': 50}
}

def init_widgets():
    widgets = {}
    pg.init()
    scrsize = np.array(pg.display.get_desktop_sizes()[0], dtype=int) * 0.9
    square_size = int(scrsize[1])
    widgets["window"] = pg.display.set_mode(scrsize)
    pg.display.set_caption("StringArtProcess")
    widgets["image_square_size"] = int(square_size)
    widgets["image_info_size"] = int(square_size) // 5
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
    widgets["scrollbars"] = [] # Initialize scrollbars list
    widgets["defaults"] = {}
    widgets["buttons"] = {}
    widgets["button_boxes"] = {}
    widgets["button_label"] = {}
    widgets["checkboxes"] = {}
    widgets["checkbox_boxes"] = {}
    widgets["checkbox_labels"] = {}
    widgets["active_box"] = None
    widgets["cursor_visible"] = True
    widgets["active_scrollbar_drag"] = None # ID of the scrollbar being dragged
    widgets["scrollbar_drag_offset_x"] = 0 # Mouse click offset from thumb's left edge

    # For storing create_config_widgets arguments for refresh
    widgets["_config_select_image_callback"] = None
    widgets["_config_submit_parameters_callback"] = None
    widgets["_config_initial_params"] = {} # Use a dict
    widgets["checkboxes_initialized"] = False # Flag for first-time setup of checkboxes
    # widgets["checkbox_param_map"] is already handled (added in create_config_widgets)

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


def create_config_widgets(widgets, select_image_callback, submit_parameters_callback,
                          initial_params_config, initial_checkboxes_config, checkbox_param_map=None):
    # Store callbacks and initial configurations for potential refresh calls
    # These are stored on the first call and then reused if subsequent calls pass None for them.
    if select_image_callback is not None:
        widgets["_config_select_image_callback"] = select_image_callback
    if submit_parameters_callback is not None:
        widgets["_config_submit_parameters_callback"] = submit_parameters_callback
    if initial_params_config is not None: # This should always be provided on first call
        widgets["_config_initial_params"] = initial_params_config

    # Use stored versions if available (e.g. for refresh)
    current_select_image_callback = widgets["_config_select_image_callback"]
    current_submit_parameters_callback = widgets["_config_submit_parameters_callback"]
    current_initial_params = widgets["_config_initial_params"]

    widgets["checkbox_param_map"] = checkbox_param_map or widgets.get("checkbox_param_map", {}) # Store or update map

    # Handle checkbox states:
    # initial_checkboxes_config is the definition of checkboxes and their default states.
    # widgets["checkboxes"] holds the current runtime state.
    if not widgets.get("checkboxes_initialized"):
        widgets["checkboxes"] = initial_checkboxes_config.copy() if initial_checkboxes_config else {}
        widgets["checkboxes_initialized"] = True
    # On refresh, initial_checkboxes_config might be the current widgets["checkboxes"] or None.
    # If it's the current widgets["checkboxes"], then this assignment is fine.
    # If it's None on refresh, we rely on widgets["checkboxes"] already being correct.
    # The key is that create_config_widgets uses widgets["checkboxes"] as the source of truth for drawing.
    if initial_checkboxes_config is not None:
         widgets["checkboxes"] = initial_checkboxes_config # Allow overriding with new config if passed

    params_to_display = current_initial_params # Use the stored/initial definition of params
    buttons = {"Select Image": current_select_image_callback, "Submit": current_submit_parameters_callback}
    # Checkboxes for UI creation loop should iterate over the *keys* defined in initial_checkboxes_config
    # or if that's not available, the current widgets["checkboxes"].
    # For now, the loop `for checkbox_name in widgets["checkboxes"].keys():` later will use current state.

    # scrollbar_param_configs is now SCROLLBAR_PARAM_CONFIGS (module level)

    widgets["parameters"] = params_to_display # This holds {name: default_value}
    widgets["buttons"] = buttons
    # widgets["checkboxes"] is already set above and holds the current true/false states.
    margin = widgets["image_square_size"] * 0.05
    # position = [margin, margin] # Original position, now labal_position will be used carefully
    labal_position = [margin, margin] # Use a mutable list for labal_position
    font = widgets["font"]

    # Calculate max label length for alignment across all configurable items
    # Use the initial config dicts for this, as they contain all possible items
    all_configurable_items = list(initial_params_config.keys()) + list(initial_checkboxes_config.keys()) # Buttons usually don't have a left-aligned label in this layout
    if not all_configurable_items:
        labal_length = 0
    else:
        # Ensure keys are strings before calling len
        labal_length = font.size(max([str(k) for k in all_configurable_items], key=len))[0]

    font_size_ref = font.size("1000000") # For height and default width calculations (using "1000000" as a reference)

    # Standard width for input elements (input boxes and scrollbar tracks)
    # Using font_size_ref[0] which is width of "1000000" for consistency, can be adjusted
    element_width = font_size_ref[0] # Base width for normal input boxes
    scrollbar_track_width = font_size_ref[0] * 1.5 # Made scrollbar tracks a bit wider
    element_height = font_size_ref[1] * 1.2 # Standard height for input elements

    if widgets["image_original"] is not None:
        widgets["image"] = widgets["image_original"].copy()
    
    widgets["labels"] = [] # Clear previous labels
    widgets["input_boxes"] = [] # Clear previous input boxes
    widgets["scrollbars"] = [] # Clear previous scrollbars
    widgets["button_boxes"] = {} # Clear previous button boxes
    widgets["button_label"] = {} # Clear previous button labels
    widgets["checkbox_boxes"] = {} # Clear previous checkbox boxes
    widgets["checkbox_labels"] = {} # Clear previous checkbox labels

    # Retrieve map and current checkbox states for conditional display logic
    active_checkbox_param_map = widgets.get("checkbox_param_map", {}) # Renamed to avoid conflict with arg
    current_checkbox_states = widgets.get("checkboxes", {}) # This is the source of truth for checkbox states

    for param_name, default_value in params_to_display.items(): # Iterate using stored initial params
        create_ui_for_param = True # Flag to determine if UI should be created

        # Check if this parameter is conditional
        is_conditional = False
        controlling_checkbox_label = None
        for chk_label, controlled_params in checkbox_param_map.items():
            if param_name in controlled_params:
                is_conditional = True
                controlling_checkbox_label = chk_label
                break

        if is_conditional:
            if not current_checkbox_states.get(controlling_checkbox_label, False):
                create_ui_for_param = False # Don't create if controlling checkbox is unchecked

        if create_ui_for_param:
            label_surface = font.render(param_name, True, (255, 255, 255))
            current_label_pos = (labal_position[0], labal_position[1] + (element_height - label_surface.get_height()) // 2)
            widgets["labels"].append((label_surface, current_label_pos))

            element_x_pos = labal_position[0] + labal_length + font_size_ref[1]

            if param_name in SCROLLBAR_PARAM_CONFIGS:
                config = SCROLLBAR_PARAM_CONFIGS[param_name]
                min_val, max_val = config['min'], config['max']
                current_val = max(min_val, min(default_value, max_val))

                track_rect = pg.Rect(
                    element_x_pos, labal_position[1],
                    scrollbar_track_width, element_height
                )
                thumb_width = 10
                thumb_x_ratio = (current_val - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
                thumb_x = track_rect.x + thumb_x_ratio * (track_rect.width - thumb_width)
                thumb_x = max(track_rect.x, min(thumb_x, track_rect.right - thumb_width))
                thumb_rect = pg.Rect(thumb_x, track_rect.y, thumb_width, track_rect.height)

                scrollbar_data = {
                    'id': param_name, 'rect': track_rect, 'thumb_rect': thumb_rect,
                    'min_val': min_val, 'max_val': max_val, 'current_val': current_val,
                }
                widgets["scrollbars"].append(scrollbar_data)
                widgets["defaults"][param_name] = default_value
            else:
                # Standard input box
                input_box_rect = pg.Rect(
                    element_x_pos, labal_position[1],
                    element_width, element_height
                )
                widgets["input_boxes"].append(input_box_rect)
                widgets["defaults"][param_name] = default_value

            labal_position[1] += element_height * 1.7 # Advance Y position only if UI was created

    # Checkboxes are created next, their Y position depends on the last `labal_position[1]`
    # The visual checkbox elements should be created based on the keys in initial_checkboxes_config
    # if available, or widgets["checkboxes"] if not.
    # For simplicity, stringart.py always passes initial_checkboxes (which becomes initial_checkboxes_config here)
    # on the first call, and on refresh it passes widgets["checkboxes"].
    # So, iterating `widgets["checkboxes"].keys()` is fine for creating the visual elements.
    for checkbox_name in widgets["checkboxes"].keys():
        # Label for checkbox (aligned with other parameter labels)
        checkbox_label_surface = font.render(checkbox_name, True, (255, 255, 255))
        clabel_pos = (labal_position[0], labal_position[1] + (element_height - checkbox_label_surface.get_height()) // 2)
        widgets["checkbox_labels"][checkbox_name] = (checkbox_label_surface, clabel_pos) # Store as tuple (surface, pos)

        # Checkbox square (aligned with input elements like input_boxes/scrollbars)
        checkbox_rect = pg.Rect(
            labal_position[0] + labal_length + font_size_ref[1], # Align with other interactive elements
            labal_position[1] + (element_height - font_size_ref[1]) // 2, # Vertically center the box part
            font_size_ref[1], # Square checkbox using reference font height
            font_size_ref[1]
        )
        widgets["checkbox_boxes"][checkbox_name] = checkbox_rect
        labal_position[1] += element_height * 1.7 # Consistent spacing

    # Buttons
    for button_name in buttons.keys():
        button_label_surface = font.render(button_name, True, (255, 255, 255))
        button_rect = pg.Rect(
            labal_position[0], # Buttons typically span more or start at the label column
            labal_position[1],
            font.size(button_name)[0] * 1.2, # Width based on text
            element_height * 1.1 # Height based on standard element height
        )
        widgets["button_boxes"][button_name] = button_rect
        widgets["button_label"][button_name] = button_label_surface # Store the surface
        labal_position[1] += element_height * 1.7 # Consistent spacing

    if widgets["input_boxes"]:
        widgets["active_box"] = widgets["input_boxes"][0]
    else:
        widgets["active_box"] = None
    # ensure 'pins' are available in widgets for drawing, populated by stringart.py
    if "pins" not in widgets: # Initialize if not present, stringart.py should fill this
        widgets["pins"] = []

    pass # No direct change here, but noting the responsibility of stringart.py


def create_information_widgets(widgets, back_callback, stop_callback, pause_callback, resume_callback):
    """
    Sets up the widgets for the information screen, including buttons for Back, Stop, Pause, and Resume.
    """
    buttons = {
        "Back": back_callback,
        "Stop": stop_callback,
        "Pause": pause_callback,
        "Resume": resume_callback,
    }
    widgets["input_boxes"] = []
    #widgets["checkboxes"] = {}
    widgets["buttons"] = buttons
    widgets["button_boxes"] = {}
    widgets["button_label"] = {}


def create_serving_widgets(widgets, back_callback, reset_callback):
    buttons = {
        "Back": back_callback,
        "Reset": reset_callback,
    }
    widgets["input_boxes"] = []
    #widgets["checkboxes"] = {}
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
        pg.draw.rect(surface, (30, 30, 30), box) # Background of input box
        pg.draw.rect(surface, (255, 255, 255), box, 2) # Border of input box

        # To get the correct param_name, we need to find which parameter this box corresponds to.
        # This assumes that the order of `widgets["input_boxes"]` matches the order of non-scrollbar params.
        input_box_param_names = [p_name for p_name in parameters.keys() if p_name not in SCROLLBAR_PARAM_CONFIGS] # Use module-level config
        if i < len(input_box_param_names):
            param_name = input_box_param_names[i]
            txt_surface = font.render(str(parameters[param_name]), True, (255, 255, 255))
            surface.blit(txt_surface, (box.x + 5, box.y + (box.height - txt_surface.get_height()) // 2)) # Centered text
            if box == widgets.get("active_box") and widgets.get("cursor_visible"):
                cursor_x = box.x + 5 + txt_surface.get_width()
                cursor_y = box.y + (box.height - txt_surface.get_height()) // 2
                pg.draw.line(
                    surface,
                    (255, 255, 255),
                    (cursor_x, cursor_y),
                    (cursor_x, cursor_y + txt_surface.get_height()),
                    2,
                )
        # else: indicates a mismatch between input_boxes and parameter names

    # Draw scrollbars
    for sb_data in widgets.get("scrollbars", []):
        # Draw track
        pg.draw.rect(surface, (50, 50, 50), sb_data['rect']) # Darker track
        pg.draw.rect(surface, (200, 200, 200), sb_data['rect'], 1) # Border for track
        # Draw thumb
        pg.draw.rect(surface, (150, 150, 150), sb_data['thumb_rect']) # Lighter thumb
        pg.draw.rect(surface, (220, 220, 220), sb_data['thumb_rect'], 1) # Border for thumb

        # Optionally, draw the current value next to the scrollbar
        val_text = str(sb_data['current_val']) # Or parameters[sb_data['id']]
        val_surface = font.render(val_text, True, (255, 255, 255))
        val_pos_x = sb_data['rect'].right + 10
        val_pos_y = sb_data['rect'].y + (sb_data['rect'].height - val_surface.get_height()) // 2
        surface.blit(val_surface, (val_pos_x, val_pos_y))

    # Updated checkbox drawing
    for checkbox_name in widgets["checkboxes"]: # Iterate by names (keys)
        checkbox_box = widgets["checkbox_boxes"][checkbox_name]
        # Retrieve the stored label surface and its position (which is a tuple: (surface, pos))
        label_surface, label_pos = widgets["checkbox_labels"][checkbox_name]

        surface.blit(label_surface, label_pos) # Draw the text label in its defined position
        pg.draw.rect(surface, (255, 255, 255), checkbox_box, 2) # Draw the checkbox square

        if widgets["checkboxes"][checkbox_name]: # If checked
            pg.draw.line(
                surface, (255, 255, 255),
                (checkbox_box.left, checkbox_box.top), (checkbox_box.right, checkbox_box.bottom), 2
            )
            pg.draw.line(
                surface, (255, 255, 255),
                (checkbox_box.left, checkbox_box.bottom), (checkbox_box.right, checkbox_box.top), 2
            )

    # Updated button drawing
    for button_name in widgets["buttons"]: # Iterate by names (keys)
        button_rect = widgets["button_boxes"][button_name]
        label_surface = widgets["button_label"][button_name] # This is the pre-rendered surface

        pg.draw.rect(surface, (255, 255, 255), button_rect, 2) # Draw button border

        # Calculate text position to center it on the button
        text_x = button_rect.x + (button_rect.width - label_surface.get_width()) // 2
        text_y = button_rect.y + (button_rect.height - label_surface.get_height()) // 2
        surface.blit(label_surface, (text_x, text_y)) # Draw the label surface


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

    image_position = [
        surface.get_width() - widgets["image_info_size"],
        surface.get_height() - widgets["image_info_size"],
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
        image_position[0] -= widgets["image_info_size"] # Adjust for next image

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
    # Define scrollbar_param_configs here or pass it or access from widgets if stored globally
    # For now, let's redefine it for clarity within handle_event, assuming it's small and fixed for this context
    # Ideally, this should be defined once globally or passed if it becomes large/dynamic
    # Using SCROLLBAR_PARAM_CONFIGS from module level now.
    # scrollbar_param_configs = SCROLLBAR_PARAM_CONFIGS # No need to redefine, use module level one.

    # Adjust mouse position once for information_surface if applicable
    # This adjustment is crucial and depends on where the event.pos is coming from (global vs surface specific)
    # Pygame events usually give global screen pos. If information_surface is not at (0,0), adjust.
    # Assuming information_surface starts at (square_size, 0) relative to the window
    # If the event is for a different surface (e.g. main image), this adjusted_pos might not be relevant.
    # For config widgets on information_surface, adjusted_pos is correct.
    # MOVED adjusted_pos CALCULATION INSIDE MOUSE EVENT TYPE CHECKS
    # REMOVED redundant adjusted_pos = ... line that was here.
    # The above comment about removal was from a previous attempt; the line is being removed NOW.

    if event.type == pg.MOUSEBUTTONDOWN:
        adjusted_pos = (event.pos[0] - square_size, event.pos[1]) if event.pos[0] >= square_size else event.pos
        # Check scrollbar interactions first, as they are more specific than generic boxes
        for sb_data in widgets.get("scrollbars", []):
            # Use adjusted_pos for collision detection with UI elements on the information_surface
            if sb_data['thumb_rect'].collidepoint(adjusted_pos):
                widgets["active_scrollbar_drag"] = sb_data['id']
                # Calculate offset of click relative to thumb's left edge, using surface-local coordinates
                widgets["scrollbar_drag_offset_x"] = adjusted_pos[0] - sb_data['thumb_rect'].x
                return # Event handled by scrollbar thumb drag start

            elif sb_data['rect'].collidepoint(adjusted_pos): # Click on track
                click_pos_relative_to_track = adjusted_pos[0] - sb_data['rect'].x
                value_ratio = click_pos_relative_to_track / sb_data['rect'].width
                new_val = sb_data['min_val'] + value_ratio * (sb_data['max_val'] - sb_data['min_val'])

                # Clamp and adjust for 'Adaptive Block'
                new_val = max(sb_data['min_val'], min(new_val, sb_data['max_val']))
                if sb_data['id'] == "Adaptive Block":
                    new_val = int(new_val)
                    if new_val < 3: new_val = 3
                    if new_val % 2 == 0: # Ensure odd
                        new_val += 1
                    new_val = max(3, min(new_val, sb_data['max_val'])) # Re-clamp if adjustment pushed it out

                sb_data['current_val'] = int(new_val)
                widgets["parameters"][sb_data['id']] = sb_data['current_val']

                # Update thumb position
                if (sb_data['max_val'] - sb_data['min_val']) == 0:
                    value_percentage = 0
                else:
                    value_percentage = (sb_data['current_val'] - sb_data['min_val']) / (sb_data['max_val'] - sb_data['min_val'])

                thumb_x = sb_data['rect'].x + value_percentage * (sb_data['rect'].width - sb_data['thumb_rect'].width)
                sb_data['thumb_rect'].x = max(sb_data['rect'].x, min(thumb_x, sb_data['rect'].right - sb_data['thumb_rect'].width))

                process_image_callback(widgets)
                return # Event handled by scrollbar track click

        for button_label in list(widgets.get("buttons", {}).keys()):
            if button_label in widgets["button_boxes"]:
                if widgets["button_boxes"][button_label].collidepoint(adjusted_pos):
                    widgets["buttons"][button_label](widgets) 
                    return

        for box in widgets.get("input_boxes", []):
            if box.collidepoint(adjusted_pos):
                widgets["active_box"] = box
                return
        
        for checkbox_label in list(widgets.get("checkboxes", {}).keys()):
            if checkbox_label in widgets["checkbox_boxes"]:
                if widgets["checkbox_boxes"][checkbox_label].collidepoint(adjusted_pos):
                    # Toggle the state
                    widgets["checkboxes"][checkbox_label] = not widgets["checkboxes"][checkbox_label]

                    # Check if this checkbox controls other parameters' visibility
                    if checkbox_label in widgets.get("checkbox_param_map", {}):
                        # Re-create/refresh the config widgets
                        create_config_widgets(
                            widgets,
                            None, # select_image_callback - use stored
                            None, # submit_parameters_callback - use stored
                            None, # initial_params_config - use stored
                            widgets["checkboxes"], # Pass current, updated checkbox states
                            widgets["checkbox_param_map"] # Pass stored map
                        )

                    # Always process image after checkbox toggle
                    process_image_callback(widgets) 
                    return

    elif event.type == pg.MOUSEMOTION:
        adjusted_pos = (event.pos[0] - square_size, event.pos[1]) if event.pos[0] >= square_size else event.pos
        if widgets.get("active_scrollbar_drag") is not None:
            active_sb_id = widgets["active_scrollbar_drag"]
            active_scrollbar = None
            for sb_data in widgets.get("scrollbars", []):
                if sb_data['id'] == active_sb_id:
                    active_scrollbar = sb_data
                    break

            if active_scrollbar:
                # Calculate new thumb_rect.x based on global mouse position and initial offset
                # The thumb should follow the mouse, constrained by the track boundaries.
                # event.pos[0] is global mouse X.
                # widgets["scrollbar_drag_offset_x"] is global X offset from thumb's left.
                # So, new thumb left edge in global coords is event.pos[0] - widgets["scrollbar_drag_offset_x"]
                # We need to convert this to be relative to the information_surface for track collision.
                # Information surface starts at x=square_size.
                # So, new_thumb_x_on_surface = (event.pos[0] - widgets["scrollbar_drag_offset_x"]) - square_size
                # No, this is simpler: thumb_rect.x is already relative to its surface.
                # The drag_offset_x was calculated from event.pos[0] (global) and thumb_rect.x (local).
                # This implies thumb_rect.x needs to be converted to global for offset calc, or event.pos[0] to local.
                # Let's assume thumb_rect.x is local to information_surface.
                # And adjusted_pos[0] is event.pos[0] made local to information_surface.
                # So, new_thumb_x_local = adjusted_pos[0] - (widgets["scrollbar_drag_offset_x"] - square_size if event.pos[0] >= square_size else widgets["scrollbar_drag_offset_x"])
                # This is getting complicated. Let's simplify:
                # active_scrollbar['thumb_rect'].x is local to information_surface.
                # event.pos[0] is global. scrollbar_drag_offset_x was event.pos[0] - active_scrollbar['thumb_rect'].x at MOUSEBUTTONDOWN.
                # This means scrollbar_drag_offset_x is the global distance from screen left to mouse, minus local distance of thumb from surface left.
                # Correct offset should be: adjusted_pos[0] - active_scrollbar['thumb_rect'].x (at time of click)
                # Let's re-evaluate offset calculation at MOUSEBUTTONDOWN:
                # widgets["scrollbar_drag_offset_x"] = adjusted_pos[0] - sb_data['thumb_rect'].x
                # This makes scrollbar_drag_offset_x the click position within the thumb, relative to thumb's left edge.

                # New thumb left edge, relative to information_surface:
                new_thumb_x_on_surface = adjusted_pos[0] - widgets["scrollbar_drag_offset_x"]

                # Clamp new_thumb_x_on_surface to be within the track boundaries
                min_thumb_x = active_scrollbar['rect'].x
                max_thumb_x = active_scrollbar['rect'].right - active_scrollbar['thumb_rect'].width
                new_thumb_x_on_surface = max(min_thumb_x, min(new_thumb_x_on_surface, max_thumb_x))

                # Calculate value based on this new thumb position
                # Ensure width of draggable area is not zero
                draggable_width = active_scrollbar['rect'].width - active_scrollbar['thumb_rect'].width
                if draggable_width <= 0:
                    value_ratio = 0
                else:
                    value_ratio = (new_thumb_x_on_surface - active_scrollbar['rect'].x) / draggable_width

                new_val = active_scrollbar['min_val'] + value_ratio * (active_scrollbar['max_val'] - active_scrollbar['min_val'])

                new_val = max(active_scrollbar['min_val'], min(new_val, active_scrollbar['max_val']))
                if active_scrollbar['id'] == "Adaptive Block":
                    new_val = int(new_val)
                    if new_val < 3: new_val = 3
                    if new_val % 2 == 0: new_val += 1
                    new_val = max(3, min(new_val, active_scrollbar['max_val']))

                active_scrollbar['current_val'] = int(new_val)
                widgets["parameters"][active_scrollbar['id']] = active_scrollbar['current_val']

                # Update thumb_rect.x based on the potentially clamped and adjusted current_val
                if (active_scrollbar['max_val'] - active_scrollbar['min_val']) == 0:
                    value_percentage_final = 0
                else:
                    value_percentage_final = (active_scrollbar['current_val'] - active_scrollbar['min_val']) / (active_scrollbar['max_val'] - active_scrollbar['min_val'])

                final_thumb_x = active_scrollbar['rect'].x + value_percentage_final * draggable_width
                active_scrollbar['thumb_rect'].x = max(active_scrollbar['rect'].x, min(final_thumb_x, active_scrollbar['rect'].right - active_scrollbar['thumb_rect'].width))

                process_image_callback(widgets)
                return # Event handled by scrollbar drag

    elif event.type == pg.MOUSEBUTTONUP:
        # adjusted_pos might be needed if handling clicks on elements during MOUSEBUTTONUP
        # For now, only resetting drag state, so not strictly needed here.
        # adjusted_pos = (event.pos[0] - square_size, event.pos[1]) if event.pos[0] >= square_size else event.pos
        if widgets.get("active_scrollbar_drag") is not None:
            widgets["active_scrollbar_drag"] = None
            return # Event handled by scrollbar drag end

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
                    calculate_pins_callback(widgets)
                    process_image_callback(widgets)
            elif event.key == pg.K_TAB:
                move_to_next_box(widgets)
            elif event.unicode.isdigit():
                current_value = str(widgets["parameters"][active_box_param_name])
                widgets["parameters"][active_box_param_name] = int(current_value + event.unicode)
                if active_box_param_name == "Number of Pins" or active_box_param_name == "Radius in Pixels":
                    # Call calculate_pins and process_image via callbacks
                    calculate_pins_callback(widgets)
                    process_image_callback(widgets)
            return # Event handled by UI input fields
    
    # If event was not handled by UI elements, pass to main event handler
    main_event_handler_callback(widgets, event)

def add_resume_button(widgets, resume_callback):
    """
    Adds a 'Resume' button to the widgets' information panel, if not already present.
    This is used when the drawing is paused, to allow resuming.
    """
    if "buttons" not in widgets:
        widgets["buttons"] = {}
    if "Resume" not in widgets["buttons"]:
        widgets["buttons"]["Resume"] = resume_callback
    # The button_boxes and button_label will be rebuilt on next draw_information call

def remove_resume_button(widgets):
    """
    Removes the 'Resume' button from the widgets' buttons dictionary, if present.
    This is used when resuming drawing from a paused state.
    """
    if "buttons" in widgets and "Resume" in widgets["buttons"]:
        widgets["buttons"].pop("Resume")

def draw_line_on_surface(surface, line_pixels, color):
    """
    Draws a sequence of points as a line on the given Pygame surface.
    line_pixels: list of (x, y) tuples.
    color: color tuple (R, G, B).
    """
    if not line_pixels or len(line_pixels) < 2:
        return
    import pygame as pg
    pg.draw.lines(surface, color, False, line_pixels, 1)
