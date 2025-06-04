import os
import json

def convert_mot_to_coco(mot_txt_files, image_dirs, output_dir, data_root, combined_json_path):
    """
    Convert multiple MOTChallenge annotation files to separate COCO-format JSON files and a combined JSON file.

    Args:
        mot_txt_files (list): List of paths to MOTChallenge ground truth .txt files.
        image_dirs (list): List of directories containing corresponding images for each sequence.
        output_dir (str): Directory to save the COCO-formatted JSON files.
        data_root (str): Root directory of the dataset.
        combined_json_path (str): Path to save the combined COCO-formatted JSON file.
    """
    # CLASSES = (
    #     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #     'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #     'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #     'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    # )
    CLASSES = (
        'plant', 'other')

    # Create categories list using the order of the COCO classes
    categories = [{"id": idx + 1, "name": class_name} for idx, class_name in enumerate(CLASSES)]

    combined_images = []
    combined_annotations = []
    image_id_offset = 0
    annotation_id_offset = 0

    for mot_txt_file, image_dir in zip(mot_txt_files, image_dirs):
        images = []
        annotations = []
        image_id = image_id_offset
        annotation_id = annotation_id_offset

        sequence_name = os.path.splitext(os.path.basename(mot_txt_file))[0]
        output_json_path = os.path.join(output_dir, f"{sequence_name}.json")

        img_files = sorted([f for f in os.listdir(data_root + image_dir) if f.endswith('.jpg') or f.endswith('.png')])

        # Create a mapping from frame number to image file
        # frame_to_image = {int(os.path.splitext(f)[0]) + 1: f for f in img_files}
        frame_to_image = {int(os.path.splitext(f)[0]): f for f in img_files}
        with open(mot_txt_file, 'r') as f:
            lines = f.readlines()
        # print(frame_to_image)
        # Process each line in the MOT .txt file
        for line in lines:
            frame, obj_id, bb_left, bb_top, bb_width, bb_height, x, y, z = map(int, line.strip().split(','))
            frame = int(frame)
            obj_id = int(obj_id)

            # Check if the frame corresponds to an image
            if frame not in frame_to_image:
                continue

            # If the image for this frame hasn't been added yet, add it
            if not any(img['file_name'].split('/')[-1] == frame_to_image[frame] for img in images):
                image_id += 1
                img_file = frame_to_image[frame]
                img_path = os.path.join(image_dir, img_file)

                # height, width = 972, 1296  # Replace with actual dimensions if available
                height, width = 1080, 810  # Replace with actual dimensions if available

                image_entry = {
                    'id': image_id,
                    'file_name': img_path,
                    'width': width,
                    'height': height,
                    'frame_id': frame
                }
                images.append(image_entry)
                combined_images.append(image_entry)

            # Create annotation
            annotation_id += 1
            annotation_entry = {
                'id': annotation_id,
                'image_id': image_id,
                # 'category_id': 59,  # Assuming 'pedestrian'; modify as needed
                'category_id': 1,  # Assuming 'pedestrian'; modify as needed
                'bbox': [int(bb_left), int(bb_top), int(bb_width), int(bb_height)],  # Convert to integer, bb_top, bb_width, bb_height],
                'area': int(bb_width) * int(bb_height),  # bb_left, bb_top, bb_width * bb_height,
                'iscrowd': 0,
                'track_id': obj_id
            }
            annotations.append(annotation_entry)
            combined_annotations.append(annotation_entry)

        # Create COCO formatted dictionary for the current sequence
        coco_format = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

        # Save to JSON file for the current sequence
        with open(output_json_path, 'w') as json_file:
            json.dump(coco_format, json_file, indent=4)

        print(f"COCO formatted annotations saved to {output_json_path}")

        # Update offsets
        image_id_offset = image_id
        annotation_id_offset = annotation_id

    # Create combined COCO formatted dictionary
    combined_coco_format = {
        'images': combined_images,
        'annotations': combined_annotations,
        'categories': categories
    }

    # Save to combined JSON file
    with open(combined_json_path, 'w') as json_file:
        json.dump(combined_coco_format, json_file, indent=4)

    print(f"Combined COCO formatted annotations saved to {combined_json_path}")

# Example usage
# mot_txt_files = [
#     './data/APPLE_MOTS/testing/instances/0006.txt',
#     './data/APPLE_MOTS/testing/instances/0007.txt',
#     './data/APPLE_MOTS/testing/instances/0008.txt',
#     './data/APPLE_MOTS/testing/instances/0010.txt',
#     './data/APPLE_MOTS/testing/instances/0011.txt',
#     './data/APPLE_MOTS/testing/instances/0012.txt'
# ]
mot_txt_files = [
    # './data/LettuceMOT/O&I1/gt/gt.txt',
    './data/LettuceMOT/straight1/gt/gt.txt',
    './data/LettuceMOT/straight3/gt/gt.txt',
    './data/LettuceMOT/straight4/gt/gt.txt',
]
# image_dirs = [
#     'testing/images/0006',
#     'testing/images/0007',
# ]
image_dirs = [
    # 'O&I1/img',
    'straight1/img',
    'straight3/img',
    'straight4/img',
]
# output_dir = './data/APPLE_MOTS/testing/json_annotations/'
# combined_json_path = './data/APPLE_MOTS/testing/annotations.json'
# data_root = "./data/APPLE_MOTS/"
output_dir = './data/LettuceMOT/json_annotations/'
combined_json_path = './data/LettuceMOT/coco_annotations_train_straight1_straight3_straight4_same_id_binary.json'
# combined_json_path = './data/LettuceMOT/coco_annotations_train_O&I1_same_id_binary.json'

data_root = "./data/LettuceMOT/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

convert_mot_to_coco(mot_txt_files, image_dirs, output_dir, data_root, combined_json_path)
