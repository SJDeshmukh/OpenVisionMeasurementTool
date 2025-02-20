import cv2
import json

def draw_city_bounding_boxes(image_path, config_path):
    # Load the image
    image = cv2.imread(image_path)

    # Load the configuration
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Find the city item
    city_item = next((item for item in config if item.get("partName") == "city"), None)

    if not city_item:
        print("City item not found in config.")
        return

    # Draw bounding boxes
    for box in city_item.get("boundingBoxes", []):
        x, y, width, height = box["x"], box["y"], box["width"], box["height"]

        # Draw rectangle (Top-left corner as reference)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the image
    cv2.imshow("City Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
draw_city_bounding_boxes("city.jpg", "config.json")