import os
import cv2
import numpy as np

def extract_bounding_boxes_and_ids(mask):
    """Extract bounding boxes and object IDs from a segmentation mask."""
    # Get unique object IDs in the mask, excluding the background (assumed to be 0)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]
    bboxes = []
    for obj_id in obj_ids:
        # Create a binary mask for the current object
        obj_mask = (mask == obj_id).astype(np.uint8)
        # Find contours of the object
        pos = np.where(obj_mask)
        xmin = np.min(pos[1]) 
        xmax = np.max(pos[1])
        ymin = np.min(pos[0]) 
        ymax = np.max(pos[0]) 
        w = xmax - xmin
        h = ymax - ymin
        object_id = (obj_id%1000)+1
        # frame_id = int(filename.split(".")[0]) + 1
        mask_h = 972
        mask_w = 1296
        # # Normalize and center bounding boxes
        # nx = (xmin + xmax) / (2 * mask_w)
        # ny = (ymin + ymax) / (2 * mask_h)
        # nw = w / mask_w
        # nh = h / mask_h
        bboxes.append((xmin, ymin, w, h, obj_id))
        # contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     # Get the bounding box coordinates for each contour
        #     x, y, w, h = cv2.boundingRect(contour)
        #     bboxes.append((x, y, w, h, obj_id))
    return bboxes

def process_sequence(sequence_path, output_path):
    """Process all frames in a sequence and save annotations in MOTChallenge format."""
    annotation_lines = []
    frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.png')])
    for frame_number, frame_file in enumerate(frame_files, start=1):
        frame_path = os.path.join(sequence_path, frame_file)
        # Read the segmentation mask
        mask = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        # Extract bounding boxes and object IDs from the mask
        bboxes = extract_bounding_boxes_and_ids(mask)
        for x, y, w, h, obj_id in bboxes:
            # Extract the instance ID from the object ID
            # instance_id = obj_id % 1000
            # Create annotation line in MOTChallenge format
            annotation_line = f"{frame_number},{obj_id},{x},{y},{w},{h},-1,-1,-1,-1"
            annotation_lines.append(annotation_line)
    # Save annotations to a text file
    with open(output_path, 'w') as f:
        f.write("\n".join(annotation_lines))

# Example usage
subsets = ["0000", "0001", "0002", "0003", "0004", "0005"]
# subsets = ["0006", "0007", "0008", "0010", "0011", "0012"]
for subset in subsets:
    sequence_path = f'./data/APPLE_MOTS/train/instances/{subset}'
    output_path = f'./data/APPLE_MOTS/train/instances/{subset}.txt'
    process_sequence(sequence_path, output_path)
