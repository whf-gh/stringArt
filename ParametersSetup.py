import pygame as pg
import numpy as np

class ParameterSetup:
    def __init__(self, size):
        pg.init()
        self.screen = pg.display.set_mode(size)
        pg.display.set_caption("Parameter Input")
        self.font = pg.font.Font(None, 36)
        self.input_boxes = []
        self.labels = []
        self.units = []
        self.parameters = {}
        self.create_widgets()
        self.active_box = self.input_boxes[0]  # Set the cursor in the first input box
        self.cursor_visible = True
        self.cursor_timer = pg.time.get_ticks()
        self.running = True

    def create_widgets(self):
        # Example parameters with units
        params = [("Number of Pins", ""), ("Max Iterations", ""), ("Radius", "cm")]
        y = 50
        for param, unit in params:
            label = self.font.render(param, True, (255, 255, 255))
            self.labels.append((label, (50, y)))
            input_box = pg.Rect(300, y, 140, 32)
            self.input_boxes.append(input_box)
            self.parameters[param] = ""
            unit_label = self.font.render(unit, True, (255, 255, 255))
            self.units.append((unit_label, (450, y)))
            y += 50

        self.submit_button = pg.Rect(300, y, 140, 32)
        self.submit_label = self.font.render("Submit", True, (255, 255, 255))

    def draw(self):
        self.screen.fill((30, 30, 30))
        for label, pos in self.labels:
            self.screen.blit(label, pos)
        for i, box in enumerate(self.input_boxes):
            pg.draw.rect(self.screen, (255, 255, 255), box, 2)
            param_name = list(self.parameters.keys())[i]
            txt_surface = self.font.render(str(self.parameters[param_name]), True, (255, 255, 255))
            self.screen.blit(txt_surface, (box.x+5, box.y+5))
            # Draw cursor if this is the active box
            if box == self.active_box and self.cursor_visible:
                cursor_x = box.x + 5 + txt_surface.get_width()
                cursor_y = box.y + 5
                pg.draw.line(self.screen, (255, 255, 255), (cursor_x, cursor_y), (cursor_x, cursor_y + txt_surface.get_height()), 2)
            # Draw unit label
            unit_label, unit_pos = self.units[i]
            self.screen.blit(unit_label, unit_pos)
        self.screen.blit(self.submit_label, (310, self.submit_button.y + 5))
        pg.draw.rect(self.screen, (255, 255, 255), self.submit_button, 2)
        pg.display.flip()

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.submit_button.collidepoint(event.pos):
                self.submit_parameters()
            for box in self.input_boxes:
                if box.collidepoint(event.pos):
                    self.active_box = box
                    break
        elif event.type == pg.KEYDOWN:
            if self.active_box:
                param_name = self.get_param_name(self.active_box)
                if event.key == pg.K_RETURN:
                    self.active_box = None
                elif event.key == pg.K_BACKSPACE:
                    self.parameters[param_name] = self.parameters[param_name][:-1]
                elif event.key == pg.K_TAB:
                    self.move_to_next_box()
                elif event.unicode.isdigit():
                    self.parameters[param_name] += event.unicode

    def move_to_next_box(self):
        if self.active_box in self.input_boxes:
            current_index = self.input_boxes.index(self.active_box)
            next_index = (current_index + 1) % (len(self.input_boxes) + 1)
            if next_index < len(self.input_boxes):
                self.active_box = self.input_boxes[next_index]
            else:
                self.active_box = None  # Move to submit button
        elif self.active_box is None:
            self.active_box = self.input_boxes[0]  # Move back to the first input box

    def get_param_name(self, box):
        index = self.input_boxes.index(box)
        return list(self.parameters.keys())[index]

    def submit_parameters(self):
        # Convert parameters to integers
        for key in self.parameters:
            if self.parameters[key] == "":
                self.parameters[key] = "0"  # Default to 0 if empty
            self.parameters[key] = int(self.parameters[key])
        print("Submitted Parameters:", self.parameters)
        self.running = False

    def loop(self):
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                self.handle_event(event)
            self.draw()
            # Blink cursor
            if pg.time.get_ticks() - self.cursor_timer > 500:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = pg.time.get_ticks()
        pg.quit()
        return self.parameters

# Example usage
if __name__ == "__main__":
    gui = ParameterSetup((800, 600))
    parameters = gui.loop()
    print("Returned Parameters:", parameters)
