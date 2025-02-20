import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from subpixel_edges import subpixel_edges

# Ensure compatibility with newer numpy versions
np.bool = bool
np.int = int

def process_image_with_subpixel(image_path, json_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

    # Load bounding box configuration from JSON
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Iterate through bounding boxes
    for item in config:
        part_name = item.get('partName', 'unknown_part')
        for box in item.get('boundingBoxes', []):
            x, y, w, h = box['x'], box['y'], box['width'], box['height']

            # Extract ROI from the image
            roi = img_gray[y:y + h, x:x + w]

            # Perform subpixel edge detection
            edges = subpixel_edges(roi, threshold=25, iters=1, order=2)

            # Draw detected subpixel edges on the image
            if hasattr(edges, 'x') and hasattr(edges, 'y'):
                for ex, ey in zip(edges.x, edges.y):
                    # Adjust coordinates to original image
                    ex, ey = int(ex + x), int(ey + y)
                    cv2.circle(img, (ex, ey), 2, (0, 0, 255), -1)  # Red dots for edges

    # Save and display the processed image
    cv2.imwrite(output_path, img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Subpixel Edge Detection")
    plt.show()

# Paths to your files
image_path = "a.jpg"
json_path = "config.json"
output_path = "output_with_edges.png"

process_image_with_subpixel(image_path, json_path, output_path)
