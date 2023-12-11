import cv2
import os
from tkinter import filedialog
from morpho import set_path



# Define global variables
drawing = False
value = 0  # Initialize the value to switch between background and foreground
brush_size = 1  # Initialize the brush size




def draw_circle(event, x, y, flags, param):
    global drawing, value, brush_size

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if value == 0:
            cv2.circle(image, (x, y), brush_size, (255, 255, 255), -1)  # Convert background to foreground
        else:
            cv2.circle(image, (x, y), brush_size, 0, -1)  # Convert foreground to background
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

if __name__ == "__main__":
    # Select the directory which contains the images to be analyzed
    root_path = filedialog.askdirectory()
    o_path = set_path(os.path.join(root_path, "Edited_images"))

    for filename in os.listdir(root_path):
        base, extension = os.path.splitext(filename)

        # Skip files with names ending in "_labeled"
        if base.endswith("_labeled"):
            continue

        # Process only files with the specified extension
        if extension.lower() not in {".jpg", ".png", ".tif"}:
            continue

        # Read the file
        input_file = os.path.join(root_path, filename)
        image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Error: Unable to load the image.")
            exit()

        cv2.namedWindow("Interactive Binary Image Editor")
        cv2.setMouseCallback("Interactive Binary Image Editor", draw_circle)

        while True:
            cv2.imshow("Interactive Binary Image Editor", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):
                value = 0  # Set the mode to convert background to foreground
            elif key == ord('b'):
                value = 1  # Set the mode to convert foreground to background
            elif ord('1') <= key <= ord('5'):
                brush_size = key - ord('0')
            elif key == 27:
                break  # Exit the program when the Esc key is pressed

        # Save the edited image
        output_path = os.path.join(o_path, f"{base}.tif")
        cv2.imwrite(output_path, image)

        # LABEL OBJECTS
        # Draw rectangles and save labeled images
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
        min_area = 2000
        max_area = 2400000
        labeled_cells = 0
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]

                # Draw a rectangle around the object in the original image
                cv2.rectangle(color_image, (bounding_box_x, bounding_box_y),
                              (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height),
                              (0, 255, 255), 2)
                labeled_cells += 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                font_scale = 0.5
                color = (0, 0, 255)
                thickness = 2
                cv2.putText(color_image, str(labeled_cells), bottom_left, font, font_scale, color, thickness)

        # Save the labeled image
        labeled_output_path = os.path.join(o_path, f"{base}_labeled.tif")
        cv2.imwrite(labeled_output_path, color_image)

        cv2.destroyAllWindows()

        print("Edited image saved at:", output_path)
        print("Labeled image saved at:", labeled_output_path)
