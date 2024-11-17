import cv2
import numpy as np
import turtle as trtl
from tkinter.messagebox import showinfo
from tkinter.filedialog import askopenfilename

class Main:
    def __init__(self):
        # Turtle fields
        self.turtle_screen = trtl.Screen()
        self.t = trtl.Turtle()
        # CV fields
        self.image: cv2.typing.MatLike = None
        self.allowed_files = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        # Methods
        self.main()
    
    """
    load_process_img
        - Loads an image into cv2 using askopenfilename
        - Makes necessary steps to prepare image for analyzation
    """
    def load_process_img(self):
        # inform user
        showinfo("Select Image", "Select the image to recreate.")
        # load the image file
        image_path = askopenfilename(filetypes=self.allowed_files)
        if not image_path:
            raise ValueError("No image file was selected. Allowed types are\n", self.allowed_files[0][1])
        # process the image provided
        self.image = cv2.imread(image_path)
        image_width, image_height = self.image.shape[:2]
        screen_width = self.turtle_screen.window_width()
        screen_height = self.turtle_screen.window_height()
        while image_width > screen_width or image_height > screen_height:
            print(image_width, image_height)
            self.image = cv2.resize(self.image, (image_width//2, image_height//2))
            image_width, image_height = self.image.shape[:2]
        # calculate the if the image is lighter or darker for bg contrast
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(grayscale)
        if mean_brightness > 127.5:
            _, _, _, max_loc = cv2.minMaxLoc(grayscale)
            original_color = tuple(self.image[max_loc[1], max_loc[0]])[::-1]
            complement_color_normalized = tuple((255-c) / 255 for c in original_color)
        else:
            _, _, min_loc, _ = cv2.minMaxLoc(grayscale)
            original_color = tuple(self.image[min_loc[1], min_loc[0]])[::-1]
            complement_color_normalized = tuple((255-c) / 255 for c in original_color)
        self.turtle_screen.bgcolor(*complement_color_normalized)
        return None

    """
    edge_detection()
        - Blurs the image for simplifying edge detection
        - Uses canny edge detection on each color channel
        - Combines the channels again using bitwise OR
    """
    def edge_detection(self) -> cv2.typing.MatLike:
        # blur the image for simplicity
        blurred_image = cv2.GaussianBlur(self.image, (5,5), 0)
        # apply canny edge detection on each color channel
        edges = []
        for i in range(3):
            edges.append(cv2.Canny(blurred_image[:, :, i], 50, 115))
        # combine edges using bitwise OR
        combined_edges = cv2.bitwise_or(edges[0], edges[1])
        combined_edges = cv2.bitwise_or(combined_edges, edges[2])
        return combined_edges

    """
    contour_detection()
        - Takes edges as an argument
        - Calculates contours from edges
    """
    def contour_detection(self, edges: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike]:
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    """
    average_color()
        - Applies a provided mask using a bitwise AND to the image
        - Calculates the mean color using cv2.mean
    """
    def average_color(self, mask) -> tuple[float]:
        masked_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        # Exclude alpha if present
        mean_color = cv2.mean(masked_image, mask=mask)[:3]
        return mean_color
    
    def main(self):
        # Set turtle speed
        self.turtle_screen.tracer(5, 0)
        # set turtle pen size
        self.t.pensize(2)
        # load and process the image
        self.load_process_img()
        # get image width and height
        image_height, image_width = self.image.shape[:2]
        # Step 1: Edge detection
        edges = self.edge_detection()
        # Step 2: Find contours and calculate average colors
        contours = self.contour_detection(edges)
        
        for contour in contours:
            # Step 3: Simplify the contour (make it have fewer points)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Step 4: Create a mask for each contour
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, thickness=cv2.FILLED)
            # Step 5: Calculate the average color of the region
            avg_color = self.average_color(mask)
            print(avg_color)
            # Step 6: Set turtle pen color to average color
            self.t.pencolor(avg_color[-1]/255, avg_color[-2]/255, avg_color[-3]/255)
            # Step 7: Draw the contour with Turtle (recreate the shape)
            self.t.penup()
            start_x, start_y = contour[0][0]
            self.t.goto(start_x - image_width/2, image_height/2 - start_y)
            self.t.pendown()
            for point in contour:
                x, y = point[0]
                self.t.goto(x - image_width/2, image_height/2 - y)
        return None

main = Main()
main.turtle_screen.mainloop()