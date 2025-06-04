import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_bounding_boxes(data_root, json_file, output_path=None):
    """
    Draw bounding boxes on images based on a COCO format JSON file.

    Parameters:
        data_root (str): Root directory containing images and annotations.
        json_file (str): Name of the COCO format JSON file.
        output_path (str, optional): Directory to save the output images with bounding boxes. If None, only displays the images.
    """
    print(data_root)
    # Construct full paths
    annotation_path = os.path.join(data_root, json_file)
    print(annotation_path)
    # Load the COCO JSON file
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Extract image information
    images = {image["id"]: image for image in coco_data["images"]}
    annotations = coco_data["annotations"]

    # Iterate over all images in the JSON file
    for image in coco_data["images"]:
        filename = image["file_name"]
        image_path = data_root + '/' + filename

        
        # Validate the constructed path
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found.")
            continue
        # Open the image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 15) if os.path.exists("arial.ttf") else None

        # Find image ID
        image_id = image['id']

        # Draw the bounding boxes for the current image
        object_count = 0
        for ann in annotations:
            if ann['image_id'] == image_id:
                bbox = ann['bbox']
                x, y, width, height = bbox
                draw.rectangle([x, y, x + width, y + height], outline="blue", width=2)

                # Draw object count on top of the bounding box
                text = f"{object_count}"
                text_position = (x, y - 10 if y - 10 > 0 else y + 5)
                draw.text(text_position, text, fill="red", font=font)
                object_count += 1
        print(f"number of objects : {object_count}")

        # Save or display the output image
        if output_path:
            output_file = os.path.join(output_path, filename)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure all parent directories exist
            img.save(output_file)
            print(f"Output saved to {output_file}")
        else:
            plt.imshow(img)
            plt.axis('off')
            plt.show()

# Example usage
data_root = "./data/APPLE_MOTS"
json_file = "testing/annotations.json"
output_path = "./lettuce_mot_log/annotated_images"

draw_bounding_boxes(data_root, json_file, output_path)
