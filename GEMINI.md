# Gemini Code Assistant Context

This document provides a comprehensive overview of the StringArt project, designed to be used as a contextual reference for the Gemini Code Assistant.

## Project Overview

StringArt is a Python application that generates intricate string art patterns from images. It features an interactive GUI built with Pygame, allowing users to load images, adjust various parameters, and see the string art being generated in real-time. The application uses a greedy algorithm to progressively build the artwork by selecting lines that best match the source image. Additionally, it can serve the generated string art data over a network for use in physical installations.

**Key Technologies:**

*   **Python 3:** The core programming language.
*   **Pygame:** For the graphical user interface and real-time rendering.
*   **Pillow (PIL):** For image manipulation tasks.
*   **OpenCV (cv2):** For advanced image processing functionalities like Canny edge detection and thresholding.
*   **Numpy:** For numerical operations, especially with image data.
*   **rembg:** For background removal from images.
*   **pytest:** For automated testing.

**Architecture:**

The application is structured into several modules:

*   `stringart.py`: The main application entry point, handling the main loop, state management, and event handling.
*   `ui.py`: Manages the Pygame-based user interface, including widgets, layouts, and drawing.
*   `image_processing.py`: Contains functions for image loading, processing (e.g., background removal, edge detection), and manipulation.
*   `string_art_generator.py`: Implements the core string art generation algorithm, including pin calculation and line selection.
*   `net_server.py`: A simple TCP server to broadcast the generated string art data.
*   `tests/`: Contains unit tests for the application's modules.

## File Structure

*   `stringart.py`: Main application file.
*   `ui.py`: Handles all UI-related logic and components.
*   `image_processing.py`: Contains all functions related to image manipulation.
*   `string_art_generator.py`: The core logic for generating the string art.
*   `net_server.py`: A TCP server for sending string art data to a client.
*   `requirements.txt`: A list of all the project's dependencies.
*   `README.md`: The project's README file.
*   `tests/`: Directory containing all the tests.
    *   `test_image_processing.py`: Tests for the `image_processing.py` module.
    *   `test_string_art_generator.py`: Tests for the `string_art_generator.py` module.

## Building and Running

**1. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the Application:**

```bash
python stringart.py
```

**3. Run Tests:**

```bash
python -m pytest tests/
```

## Development Conventions

*   **Modularity:** The code is organized into modules with specific responsibilities (UI, image processing, string art generation).
*   **Callbacks:** The main application (`stringart.py`) uses callbacks to connect UI events (from `ui.py`) to the corresponding logic in other modules.
*   **State Management:** The application's state is managed in a central `widgets` dictionary, which is passed to the different modules.
*   **Testing:** The project uses `pytest` for unit testing. Tests are located in the `tests/` directory and are organized by module.

## Key Workflows

### Generating String Art

1.  Run the application: `python stringart.py`.
2.  The main window opens, showing the configuration panel.
3.  Click the "Select Image" button to load an image.
4.  Adjust the parameters in the configuration panel (e.g., number of pins, number of lines).
5.  Click the "Submit" button to start the string art generation process.
6.  The application will switch to the drawing view, where you can see the string art being generated in real-time.

### Using the Network Server

1.  During the string art generation, press the 'N' key to enable network sending mode.
2.  Once the drawing process is stopped, the application will automatically start a TCP server.
3.  A client application can then connect to the server (default: `localhost:9999`) to receive the string art data.
