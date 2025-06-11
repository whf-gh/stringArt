# StringArt

![](https://github.com/whf-gh/StringArt/blob/main/StringArtMeruem.gif)
![](https://github.com/whf-gh/StringArt/blob/main/Meruem.png)

## Introduction

StringArt is a Python application that generates intricate string art patterns from images. It uses a greedy algorithm to progressively build the artwork by selecting lines that best match the source image. The application features an interactive GUI built with Pygame, allowing users to load images, adjust parameters, and observe the string art generation in real-time. Additionally, it can serve the generated string art data over a network for use in physical installations.

## Features

The `stringart.py` application offers a comprehensive set of features for generating and interacting with string art:

*   **Image Loading:** Easily load images using a file dialog.
*   **Image Preprocessing:**
    *   Background Removal: Automatically remove the background from images.
    *   Denoising: Reduce noise in the input image for clearer patterns.
    *   Edge Detection: Apply Canny edge detection to highlight contours.
    *   Thresholding: Options for adaptive and global thresholding to convert images to black and white.
    *   Line Thickness Adjustment: Modify the thickness of lines in the processed image.
*   **Circular Mask:** Apply a circular mask to the image, focusing the string art generation within a defined circular area.
*   **Configurable Parameters:**
    *   Number of Pins: Define the number of pins around the circular frame.
    *   Number of Lines: Set the maximum number of lines to be drawn.
    *   Radius in Pixels: Specify the radius of the string art canvas in pixels.
    *   Radius in millimeters: Define the real-world radius of the physical string art.
    *   Minimum Pins Line Spans: Set the minimum number of pins for a line can spans.
    *   Max Pin Usage: Limit the number of times a single pin can be used.
*   **Interactive GUI:**
    *   Built with Pygame for a responsive user experience.
    *   Real-time display of the generated string art.
*   **Information Panel:**
    *   Displays statistics such as:
        *   Diameter (mm)
        *   Number of Nails (Pins)
        *   Number of Lines
        *   Total Length of String (meters)
*   **Drawing Process Control:**
    *   Pause, resume, or stop the string art generation process at any time.
*   **Network Serving:**
    *   Serves the generated string art data (pin indices or real-world coordinates) over a TCP network connection. The integrated `net_server.py` functionality is automatically started by the main application when drawing is stopped if the network option was enabled during the process.

## Algorithm

The core of `stringart.py` employs a **greedy algorithm** to construct the string art. At each step, the algorithm evaluates a random subset of potential next pins and selects the line (connecting the current pin to one of the pins in the subset) that results in the greatest reduction of darkness in the target image along its path. This iterative process continues until the desired number of lines is drawn or other stopping criteria are met. While the selection of a subset of pins for evaluation introduces an element of randomness, the decision-making process at each step is deterministic based on the "darkest line" heuristic.

## Dependencies

The project relies on the following Python libraries:

*   `pygame`: For the graphical user interface and real-time rendering.
*   `rembg`: For background removal from images.
*   `Pillow` (PIL): For image manipulation tasks.
*   `numpy`: For numerical operations, especially with image data.
*   `opencv-python` (cv2): For advanced image processing functionalities like Canny edge detection and thresholding.
*   `tkinter`: For the file dialog functionality (usually included with standard Python distributions).

## Installation

1.  **Clone the repository (if applicable) or download the source files.**
2.  **Install the required dependencies using pip:**

    ```bash
    pip install pygame rembg Pillow numpy opencv-python
    ```
    *Note: `tkinter` is typically included with Python. If it's missing, you might need to install it separately depending on your Python distribution and operating system (e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu).*

## Running the Application

1.  **Navigate to the directory containing the `stringart.py` file.**
2.  **Run the application from your terminal:**

    ```bash
    python stringart.py
    ```

## Network Server (`net_server.py` functionality)

The application includes functionality, originally in `net_server.py`, to act as a simple TCP server. This server works in conjunction with the main `stringart.py` process. When the string art generation is stopped and the network option (activated by pressing 'N' during generation) is enabled, `stringart.py` automatically starts this server functionality. It then transmits the sequence of pin connections (either as pin indices or real-world coordinates) to any connected client.

**Purpose:**

*   To decouple the string art generation from its physical reproduction.
*   To allow other applications or devices (e.g., a robot arm controller) to consume the string art data over the network.

**How it works:**

1.  During the string art generation in `stringart.py`, press 'N' to enable network sending mode.
2.  Once the drawing process is stopped (either by completion or manually), if network sending was enabled, the application will automatically start listening for a client connection on a predefined host and port (default: `localhost:9999`).
3.  A client application can then connect to this address to receive the string art data. The data is printed to the server's console and sent to the client. This functionality can be modified to save the data to a file or forward it to another process.

This integrated approach means you do not need to run a separate `net_server.py` script. The main application handles the serving of data when configured to do so.

## Running Tests

To run the automated tests, ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt
```
Then, navigate to the root directory of the project and run pytest:
```bash
python3 -m pytest tests/
```
This will discover and run all tests in the `tests` directory.
