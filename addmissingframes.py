import json

# Load the JSON file
with open('/home/amir/workspace/DuViDA/data/LettuceMOT/coco_annotations_train_O&I2.json', 'r') as file:
    data = json.load(file)

# Add missing frames to the "images" section
for frame_id in range(393, 700):
    image_entry = {
        "id": frame_id,
        "file_name": f"O&I2/img/{frame_id:06d}.png",
        "height": 1080,
        "width": 810
    }
    data["images"].append(image_entry)

# Sort the images by ID to maintain order
data["images"].sort(key=lambda x: x["id"])

# Save the updated JSON file
with open('/home/amir/workspace/DuViDA/data/LettuceMOT/coco_annotations_train_O&I2_.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Missing frames added and JSON file updated.")
