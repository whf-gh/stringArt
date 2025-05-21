import pygame as pg
from enum import Enum
import numpy as np
import copy # For deepcopying parts of widgets if necessary

# New module imports
import ui
import image_processing
import string_art_generator
from net_server import TCPServer as Server

# PIL and Tkinter should ideally be encapsulated within image_processing or ui
# For image conversion, PIL might be used here temporarily before passing to ui
from PIL import Image # For converting numpy array to PIL then to Pygame surface

class State(Enum):
    CONFIGURING = 1
    DRAWING = 2
    PAUSING = 3
    SERVING = 4
    QUITING = 5

# --- Callback Functions ---
def pil_to_pygame_surface(pil_image):
    """Converts a PIL image to a Pygame surface."""
    if pil_image is None:
        return None
    # Ensure image is in RGB or RGBA for Pygame
    if pil_image.mode not in ('RGB', 'RGBA'):
        pil_image = pil_image.convert('RGB')
    
    return pg.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode)

def handle_recalculate_pins_and_process_image(widgets):
    """Recalculates pins and re-processes the image. Called when relevant params change."""
    # Recalculate pins
    if "parameters" in widgets and "image_square_size" in widgets:
        widgets["pins"] = string_art_generator.calculate_pins(
            widgets["image_square_size"],
            widgets["parameters"].get("Radius in Pixels", widgets["image_square_size"] // 4),
            widgets["parameters"].get("Number of Pins", 200)
        )
    # Re-process image
    handle_process_image(widgets)

def handle_select_image(widgets):
    """Callback for selecting an image."""
    pil_img, pil_img_original = image_processing.select_image(widgets["image_square_size"])
    if pil_img and pil_img_original:
        widgets["image_original_pil"] = pil_img_original # Store original PIL for re-processing
        widgets["image_pil"] = pil_img # Store current PIL for processing
        
        # Initial processing based on current checkbox states
        handle_process_image(widgets)
    # ui.draw_preview_image and ui.draw_config_widgets will be called in the main loop

def handle_process_image(widgets):
    """Callback for processing the image based on checkbox states."""
    if widgets.get("image_pil") is None: # Or image_original_pil if that's the source
        # Try to get from original if current image_pil is not set
        if widgets.get("image_original_pil") is None:
            print("No image selected to process.")
            return
        else: # Use original as the base for processing
             widgets["image_pil"] = widgets["image_original_pil"].copy()


    # Ensure 'init_array' from original image is available if needed by process_image
    # This depends on how process_image is structured. If it needs an initial array from
    # the originally selected image (before other processing steps), ensure it's passed.
    # For now, assuming process_image uses widgets["image_pil"] (a PIL image) as its starting point.
    # And that image_original_pil holds the true original if needed for a full reset.
    
    # The process_image function in image_processing.py expects:
    # image_to_process (PIL), checkboxes_state, image_square_size, radius_pixels, existing_init_array (optional)
    # Let's assume image_to_process is the currently selected (and potentially resized/cropped) image
    
    # If image_original_pil exists, use it as the pristine source for processing
    source_pil_image = widgets.get("image_original_pil", widgets.get("image_pil"))
    if not source_pil_image:
        print("No source image available for processing.")
        return

    # Convert original PIL to numpy array for 'existing_init_array' if needed
    existing_init_array_for_processing = np.array(source_pil_image.convert('L'))


    processed_pil_image, init_array_np, processed_array_np = image_processing.process_image(
        source_pil_image.copy(), # Pass a copy to avoid modifying the stored original PIL
        widgets["checkboxes"],
        widgets["image_square_size"],
        widgets["parameters"].get("Radius in Pixels", widgets["image_square_size"] // 4),
        existing_init_array=existing_init_array_for_processing 
    )

    widgets["image_pil"] = processed_pil_image # Update with the newly processed PIL image
    widgets["init_array"] = init_array_np     # Main array for string algorithm
    widgets["processed_array"] = processed_array_np # Secondary array (e.g. edges)

    # Update Pygame surfaces for UI display
    widgets["image"] = pil_to_pygame_surface(widgets["image_pil"])
    if widgets["init_array"] is not None:
        widgets["display_init_surface"] = pil_to_pygame_surface(Image.fromarray(widgets["init_array"]))
    else:
        widgets["display_init_surface"] = None
    if widgets["processed_array"] is not None:
        widgets["display_processed_surface"] = pil_to_pygame_surface(Image.fromarray(widgets["processed_array"]))
    else:
        widgets["display_processed_surface"] = None
        
    # Update use_pin_index based on checkbox
    widgets["use_pin_index"] = widgets["checkboxes"].get("Use Pin Index", True)


def handle_submit_parameters(widgets):
    """Callback for submitting parameters from the config UI."""
    if widgets.get("image_pil") is None: # Check if an image has been selected and processed
        print("Please select an image first.")
        # Optionally, provide user feedback through the UI if possible
        return

    # Parameters are already updated in widgets["parameters"] by ui.handle_event's text input logic
    # No, ui.handle_event only updates the string value. Conversion to int happens here.
    for key, value_str in widgets["parameters"].items():
        try:
            widgets["parameters"][key] = int(value_str)
        except ValueError:
            print(f"Warning: Could not convert parameter {key} value '{value_str}' to int. Using default or 0.")
            # Fallback to default if available, or 0. The ui.py should provide defaults initially.
            widgets["parameters"][key] = widgets["defaults"].get(key, 0)


    # If init_array is not yet set (e.g. first submission after image selection and processing)
    # it should have been set by handle_process_image.
    # If image_pil exists but init_array somehow isn't there, create from image_pil
    if widgets["init_array"] is None and widgets.get("image_pil"):
        print("Warning: init_array was None after image processing. Re-creating from current PIL image.")
        widgets["init_array"] = np.array(widgets["image_pil"].convert('L'))
    elif widgets["init_array"] is None:
        print("Error: No image data available (init_array is None). Cannot start drawing.")
        return


    widgets["state"] = State.DRAWING
    
    # Reset drawing state
    widgets["current_index"] = 0 # Start from the first pin (or a random one)
    widgets["last_index"] = None
    widgets["steps"] = [] # Store pin coordinates
    widgets["steps_index"] = [] # Store pin indices
    widgets["total_length_in_pixels"] = 0
    
    # Initial pin for starting the drawing process
    if widgets["pins"]: # Ensure pins are calculated
        widgets["steps"].append(widgets["pins"][widgets["current_index"]])
        widgets["steps_index"].append(widgets["current_index"])
    else:
        print("Error: Pins not calculated. Cannot start drawing.")
        widgets["state"] = State.CONFIGURING # Revert to config
        return

    # Setup UI for drawing information
    ui.create_information_widgets(widgets, 
                                   back_callback=handle_back_to_config, 
                                   stop_callback=handle_stop_drawing, 
                                   pause_callback=handle_pause_drawing, 
                                   resume_callback=handle_resume_drawing)
    # Prepare the drawing surface in the UI
    ui.init_drawing(widgets) # This clears the surface and draws pins/circle
    
    # Remove "Resume" button if it exists from a previous pause
    if "Resume" in widgets["buttons"]:
        widgets["buttons"].pop("Resume")
        # ui.py should also remove it from its button_boxes/labels if create_information_widgets doesn't fully reset them.
        # For simplicity, assume create_information_widgets correctly rebuilds the button list.


def handle_main_event(widgets, event):
    """Callback for global events not handled by specific UI elements."""
    if event.type == pg.QUIT:
        widgets["state"] = State.QUITING
    # Add other global event handling here if needed

def handle_stop_drawing(widgets):
    """Callback to stop drawing and enter serving mode."""
    widgets["state"] = State.SERVING
    data_list = []
    if widgets["use_pin_index"]:
        data_list = widgets["steps_index"]
    else:
        if not widgets["steps"]: # Ensure steps exist
             print("No steps recorded to send to server.")
        else:
            data_list = string_art_generator.to_real_coordinates(
                widgets["steps"],
                widgets["image_square_size"],
                widgets["parameters"].get("Radius in Pixels", 1), # Avoid div by zero
                widgets["parameters"].get("Radius in milimeter", 0)
            )
            
    widgets["server"] = Server(data_list, "0.0.0.0", 65432)
    ui.create_serving_widgets(widgets, 
                              back_callback=lambda w: ui.create_information_widgets(w, handle_back_to_config, handle_stop_drawing, handle_pause_drawing, handle_resume_drawing), # Go back to info screen
                              reset_callback=handle_reset_to_config) # Reset could go to full config

def handle_reset_to_config(widgets):
    """Resets state to configuration, clears server and drawing progress."""
    if widgets.get("server"):
        widgets["server"].stop()
        widgets["server"] = None
    widgets["steps"] = []
    widgets["steps_index"] = []
    widgets["total_length_in_pixels"] = 0
    widgets["current_index"] = 0
    widgets["last_index"] = None
    # Potentially reset image arrays if needed, or keep them for user
    # widgets["image_pil"] = None
    # widgets["image_original_pil"] = None
    # widgets["image"] = None 
    # widgets["init_array"] = None
    # widgets["processed_array"] = None
    
    widgets["state"] = State.CONFIGURING
    # Re-initialize config widgets with default parameters and callbacks
    _setup_config_ui(widgets)


def handle_pause_drawing(widgets):
    """Callback to pause drawing."""
    if widgets["state"] == State.DRAWING: # Only pause if currently drawing
        widgets["state"] = State.PAUSING
        # ui.create_information_widgets should ideally handle adding/removing Resume button
        # For now, let's assume it's handled by redrawing the info panel or specific button logic in ui.py
        # If not, we might need: ui.add_resume_button(widgets, handle_resume_drawing)

def handle_resume_drawing(widgets):
    """Callback to resume drawing."""
    if widgets["state"] == State.PAUSING: # Only resume if paused
        widgets["state"] = State.DRAWING
        # ui.create_information_widgets should handle removing Resume and ensuring Pause is there
        # If not, we might need: ui.remove_resume_button(widgets)


def handle_back_to_config(widgets):
    """Callback to go back to the configuration screen from information/serving."""
    if widgets.get("server"): # Stop server if running
        widgets["server"].stop()
        widgets["server"] = None
    widgets["state"] = State.CONFIGURING
    _setup_config_ui(widgets)


def _setup_config_ui(widgets):
    """Helper to initialize or re-initialize config UI elements and parameters."""
    # Define initial parameters for the UI
    # These will be used by ui.create_config_widgets
    # The actual values in widgets["parameters"] will be stringified by ui.py for display
    # And converted back to int by handle_submit_parameters or when text input changes.
    initial_params = {
        "Number of Pins": widgets["parameters"].get("Number of Pins", 200),
        "Number of Lines": widgets["parameters"].get("Number of Lines", 500),
        "Radius in Pixels": widgets["parameters"].get("Radius in Pixels", widgets["image_square_size"] // 2),
        "Radius in milimeter": widgets["parameters"].get("Radius in milimeter", 190),
        "Shortest Line in Pixels": widgets["parameters"].get("Shortest Line in Pixels", widgets["image_square_size"] // 10),
        "Max Pin Usage": widgets["parameters"].get("Max Pin Usage", 15),
    }
    widgets["parameters"] = {k: str(v) for k,v in initial_params.items()} # UI expects strings for input boxes
    widgets["defaults"] = initial_params # Store defaults for reset or validation

    initial_checkboxes = {
        "Remove Background": widgets["checkboxes"].get("Remove Background", False),
        "denoise": widgets["checkboxes"].get("denoise", False),
        "Canny edge detection": widgets["checkboxes"].get("Canny edge detection", False),
        "Adaptive thresholding": widgets["checkboxes"].get("Adaptive thresholding", False),
        "Global Thresholding": widgets["checkboxes"].get("Global Thresholding", False),
        "Line Thickness": widgets["checkboxes"].get("Line Thickness", False),
        "Use Processed Image Only": widgets["checkboxes"].get("Use Processed Image Only", False),
        "Use Pin Index": widgets["checkboxes"].get("Use Pin Index", True),
    }
    widgets["checkboxes"] = initial_checkboxes
    
    # Initial pin calculation
    widgets["pins"] = string_art_generator.calculate_pins(
        widgets["image_square_size"],
        initial_params["Radius in Pixels"], # Use actual int value
        initial_params["Number of Pins"]    # Use actual int value
    )
    
    ui.create_config_widgets(
        widgets,
        select_image_callback=handle_select_image,
        submit_parameters_callback=handle_submit_parameters,
        initial_params=initial_params, # Pass the numeric initial params
        initial_checkboxes=initial_checkboxes,
        # calculate_pins_callback is not directly used by ui.create_config_widgets anymore for setting pins,
        # but could be a generic callback if UI needed to trigger recalculation.
        # For now, stringart.py handles pin calculation initially and on param changes.
        calculate_pins_callback=lambda w: string_art_generator.calculate_pins(
            w["image_square_size"], 
            int(w["parameters"]["Radius in Pixels"]), 
            int(w["parameters"]["Number of Pins"])
        ), # This might be used by ui.handle_event
        process_image_callback=handle_process_image # This is used by ui.handle_event for checkbox changes
    )

# --- Main Application Setup ---
def init_app_widgets():
    """Initializes the main application widgets dictionary."""
    # Initialize UI specific parts first
    widgets = ui.init_widgets() # This sets up surfaces, font, basic UI structure from ui.py

    # Application specific state and default parameters
    widgets["state"] = State.CONFIGURING
    widgets["image_pil"] = None # Holds the current PIL image for processing/display
    widgets["image_original_pil"] = None # Holds the original selected PIL image
    
    # Pygame surfaces for display - will be populated after image processing
    widgets["image"] = None # Main preview (pygame.Surface)
    widgets["display_init_surface"] = None # Thumbnail for init_array (pygame.Surface)
    widgets["display_processed_surface"] = None # Thumbnail for processed_array (pygame.Surface)

    widgets["init_array"] = None # Numpy array for string algorithm
    widgets["processed_array"] = None # Numpy array for edges/visuals
    
    widgets["decay"] = 1.5 # Example: if needed by update_image_array logic (currently not passed)
    widgets["current_index"] = 0
    widgets["last_index"] = None
    widgets["pins"] = []
    widgets["use_pin_index"] = True # Default, will be updated by checkbox
    widgets["steps"] = [] # List of (x,y) pin coordinates for string path
    widgets["steps_index"] = [] # List of pin indices for string path
    widgets["total_length_in_pixels"] = 0
    widgets["server"] = None

    # Initialize parameters and checkboxes with defaults
    # These are set up in _setup_config_ui, which is called next in main()
    widgets["parameters"] = {} 
    widgets["checkboxes"] = {}
    widgets["defaults"] = {}


    # Initial call to setup config UI specific parameters and checkboxes
    # This will populate widgets["parameters"], widgets["checkboxes"], widgets["pins"]
    # and then call ui.create_config_widgets
    # _setup_config_ui(widgets) # Called from main after widgets is created

    return widgets

def main():
    pg.init()
    widgets = init_app_widgets() # Initialize all widget data
    
    # This will populate parameters, checkboxes, pins, and then call ui.create_config_widgets
    _setup_config_ui(widgets) 
                                 
    screen = widgets["window"]
    string_drawing_surf = widgets["string_drawing_surface"]
    info_surf = widgets["information_surface"]
    square_size = widgets["image_square_size"]

    clock = pg.time.Clock()
    cursor_blink_timer = 0
    blink_interval = 500  # milliseconds

    while widgets["state"] != State.QUITING:
        dt = clock.tick(60) # Aim for 60 FPS
        cursor_blink_timer += dt
        if cursor_blink_timer >= blink_interval:
            widgets["cursor_visible"] = not widgets["cursor_visible"]
            cursor_blink_timer %= blink_interval

        for event in pg.event.get():
            # Pass event to UI handler first. It will call handle_main_event if not processed.
            # ui.handle_event needs the calculate_pins_callback and process_image_callback
            # for when parameter text inputs or checkboxes are changed.
            ui.handle_event(widgets, event, 
                            process_image_callback=handle_process_image,
                            calculate_pins_callback=handle_recalculate_pins_and_process_image, # For when num pins/radius changes
                            main_event_handler_callback=handle_main_event)

        if widgets["state"] == State.SERVING and widgets["server"] is not None:
            widgets["server"].handle_events()

        # --- Drawing logic based on state ---
        screen.fill((50, 50, 50)) # Background for the whole window

        match widgets["state"]:
            case State.CONFIGURING:
                ui.draw_config_widgets(widgets)
                ui.draw_preview_image(widgets) # Uses widgets["image"] (pygame surface) and widgets["pins"]
            case State.DRAWING:
                # Calculate next line segment
                next_pin_coord, line_pixels, next_pin_idx = string_art_generator.generate_next_string_segment(widgets)
                
                if next_pin_coord and line_pixels and next_pin_idx is not None:
                    # Update state based on new segment
                    widgets["last_index"] = widgets["current_index"]
                    widgets["current_index"] = next_pin_idx
                    widgets["steps"].append(next_pin_coord)
                    widgets["steps_index"].append(next_pin_idx)
                    widgets["total_length_in_pixels"] += len(line_pixels)
                    
                    # Draw the line segment on the string_drawing_surface
                    # Assuming ui.py has a function like this or adapt ui.init_drawing or a new one.
                    # For now, directly draw using pg.draw.lines for simplicity if line_pixels is a list of points.
                    # Or, if draw_line from original stringart is now in ui.py:
                    if hasattr(ui, 'draw_line_on_surface'): # Check if helper exists
                        ui.draw_line_on_surface(string_drawing_surf, line_pixels, widgets["color_black"])
                    else: # Fallback to basic pygame line drawing for each segment
                        if len(line_pixels) > 1: # Need at least 2 points for lines
                             # pg.draw.lines(string_drawing_surf, widgets["color_black"], False, line_pixels, 1)
                             # The line_pixels from calculate_line_darkness is a list of individual pixels.
                             # So, we need to draw them point by point.
                             for p_x, p_y in line_pixels:
                                if 0 <= p_x < square_size and 0 <= p_y < square_size:
                                     string_drawing_surf.set_at((p_x, p_y), widgets["color_black"])
                else:
                    # No more lines to draw or error
                    print("Drawing finished or no suitable next pin found.")
                    handle_stop_drawing(widgets) # Transition to serving or allow reset

                # Check for pause condition (e.g., max lines)
                # Original had: if (len(widgets["steps"]) - 1) % widgets["parameters"].get("Number of Lines", float('inf')) == 0:
                # This implies Number of Lines is a batch size for pausing.
                # Let's use a more direct interpretation if Number of Lines means total lines.
                if (len(widgets["steps"]) -1) >= widgets["parameters"].get("Number of Lines", float('inf')):
                    print(f"Reached target number of lines: {widgets['parameters']['Number of Lines']}.")
                    handle_stop_drawing(widgets) # Or handle_pause_drawing(widgets) if it's a batch
                
                # Update and draw information panel
                ui.draw_information(widgets)

            case State.PAUSING:
                # Simply draw the information panel, string surface is static
                ui.draw_information(widgets)
            case State.SERVING:
                ui.draw_serving(widgets)
            case State.QUITING:
                pass # Loop will terminate

        # Blit surfaces to the main screen
        if widgets["state"] != State.QUITING:
            if widgets["state"] == State.CONFIGURING or widgets["state"] == State.DRAWING or widgets["state"] == State.PAUSING:
                screen.blit(string_drawing_surf, (0, 0))
            
            # Information surface is common to all states except maybe quiting fully
            screen.blit(info_surf, (square_size, 0))
            pg.display.flip()

    # Cleanup
    if widgets.get("server"):
        widgets["server"].stop()
    pg.quit()

if __name__ == "__main__":
    main()
