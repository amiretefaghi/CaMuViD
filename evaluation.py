import torch
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from scipy.spatial import KDTree
import numpy as np
from sklearn.cluster import DBSCAN
from torchvision.ops import box_iou  # Import the box_iou function
import matplotlib.pyplot as plt

# IoU Calculation using tensors
def calculate_iou_tensor(box1, box2):
    """Calculates the IoU between two bounding boxes in tensor format."""
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate areas of the bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# IoU Matrix Calculation for tensors
def calculate_iou_matrix_tensor(pred_boxes, gt_boxes):
    """Calculates the IoU matrix between predicted and ground truth boxes in tensor format."""
    iou_matrix = torch.zeros((pred_boxes.size(0), gt_boxes.size(0)))

    for i in range(pred_boxes.size(0)):
        for j in range(gt_boxes.size(0)):
            iou_matrix[i, j] = calculate_iou_tensor(pred_boxes[i], gt_boxes[j])
    
    return iou_matrix

# Performance Evaluation with Hungarian Algorithm using tensors
def evaluate_detection_with_hungarian(pred_boxes, gt_boxes, gt_ids, iou_threshold=0.45):
    # Step 1: Compute IoU matrix

    # Handle the case where there are no detected bounding boxes
    if len(pred_boxes) == 0:
        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_boxes)  # All ground-truth boxes are considered false negatives

        precision = 0
        recall = 0
        f1_score = 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'matched_gt_ids': [],
            'pred_to_gt_map': {},
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'iou_per_gt_id': {},
            'iou_matrix': None,
            'row_ind': [],
            'col_ind': [],
            'unmatched_pred_boxes': [],
            'unmatched_pred_boxes_': [],
            'unmatched_pred_indices': [],
            'unmatched_pred_iou': {}
        }
    
    # iou_matrix = calculate_iou_matrix_tensor(pred_boxes, gt_boxes)
    iou_matrix = box_iou(pred_boxes[:,:4], gt_boxes)  # Efficiently calculates the IoU matrix

    # Step 2: Use the Hungarian algorithm to find optimal assignment
    # Convert IoU matrix to NumPy array for Hungarian algorithm
    iou_matrix_np = iou_matrix.cpu().numpy()

    # The algorithm minimizes the cost, so we use negative IoU to maximize the assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_np)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_gt_ids = []
    pred_to_gt_map = {}
    iou_per_gt_id = {}  # To store IoU of each matched GT ID

    # Keep track of unmatched predicted boxes
    matched_pred_indices = set()

    for i, j in zip(row_ind, col_ind):
        if iou_matrix_np[i, j] >= iou_threshold:  # Check if the IoU is greater than or equal to the threshold
            true_positives += 1
            matched_gt_ids.append(gt_ids[j].item())  # Convert tensor ID to a Python scalar
            pred_to_gt_map[i] = gt_ids[j].item()
            matched_pred_indices.add(i)
            iou_per_gt_id[gt_ids[j].item()] = iou_matrix_np[i, j]
        else:
            false_positives += 1
    unmatched_pred_indices = set(range(len(pred_boxes))) - matched_pred_indices
    unmatched_pred_boxes = pred_boxes[list(unmatched_pred_indices)]  # Get the unmatched predicted boxes
    unmatched_pred_boxes_ = pred_boxes[list(unmatched_pred_indices)]  # Get the unmatched predicted boxes
    # print(unmatched_pred_boxes)
    unmatched_pred_boxes_[:,2] = unmatched_pred_boxes_[:,2] - unmatched_pred_boxes_[:,0]
    unmatched_pred_boxes_[:,3] = unmatched_pred_boxes_[:,3] - unmatched_pred_boxes_[:,1]
    # Calculate the best IoU for unmatched predicted boxes (even if below the threshold)
    iou_unmatched_pred = {}
    for unmatched_idx in unmatched_pred_indices:
        # Find the maximum IoU for this unmatched prediction with any GT box
        max_iou = np.max(iou_matrix_np[unmatched_idx, :])
        iou_unmatched_pred[unmatched_idx] = max_iou

    false_negatives = len(gt_boxes) - true_positives
    false_positives += len(pred_boxes) - true_positives

    # Precision, Recall, F1-Score Calculation
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matched_gt_ids': matched_gt_ids,
        'pred_to_gt_map': pred_to_gt_map,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'iou_per_gt_id': iou_per_gt_id,  # Add IoU for each detected GT ID,
        'iou_matrix': iou_matrix,
        'row_ind': row_ind,
        'col_ind': col_ind,
        'unmatched_pred_boxes': unmatched_pred_boxes,
        'unmatched_pred_boxes_': unmatched_pred_boxes_,
        'unmatched_pred_indices': list(unmatched_pred_indices),
        'unmatched_pred_iou': iou_unmatched_pred
    }

def project_unmatched_boxes_to_world(unmatched_boxes, dataset, cam_id):
    """
    Project the center bottom point of unmatched bounding boxes to world coordinates.
    Args:
        unmatched_boxes: List of unmatched bounding boxes for a particular view.
        cam_id: ID of the camera/view.
    Returns:
        List of world coordinates for the unmatched bounding boxes.
    """
    world_coords = []
    for box in unmatched_boxes:
        # Convert the bounding box to a numpy array if it's a tensor (move to CPU first)
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
    
        # Calculate the center bottom point of the bounding box
        center_bottom_x = (box[0] + box[2]) / 2
        center_bottom_y = box[3]  # Use the bottom y-coordinate

        # Create the image coordinate from the center bottom point
        image_coord = np.array([[center_bottom_x], [center_bottom_y]])
        
        # Project the center bottom point to world coordinates
        world_coord = dataset.get_worldcoord_from_imagecoord(image_coord, cam_id)
        world_grid = dataset.get_worldgrid_from_worldcoord(world_coord)
        world_coords.append(world_grid)
    
    return world_coords

def merge_close_points(world_coords_list, merge_threshold=100):
    """
    Merge close world coordinates from different views using a KD-Tree.
    Args:
        world_coords_list: List of world coordinates from different views.
        merge_threshold: Threshold distance to consider two points as the same.
    Returns:
        List of merged points.
    """
    all_coords = np.concatenate(world_coords_list, axis=0)  # Combine and transpose to have (N, 2)
    
    # Apply DBSCAN to cluster points that are within the merge_threshold
    db = DBSCAN(eps=merge_threshold, min_samples=1).fit(all_coords)

    # Get the cluster labels for each point
    labels = db.labels_

    # Group points by their cluster and compute the centroid of each cluster
    merged_points = []
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_points = all_coords[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        merged_points.append(centroid)

    return np.array(merged_points)

def aggregate_detections_across_views(view_matches, all_gt_ids):
    """Aggregates the detections across all views for each frame and includes all ground truth IDs.
    
    Args:
        view_matches: A list of matched GT IDs from each view for a single frame.
                      Each element of the list corresponds to the result of matched GT IDs for one view.
        all_gt_ids: A set of all ground truth IDs that should be included, even if unmatched.
    
    Returns:
        A dictionary with each GT ID and the number of times it was detected across all views.
    """
    detection_count = defaultdict(int)
    
    # Loop through matches from each view and count how often each ID is detected
    for matches in view_matches:
        for gt_id in matches:
            detection_count[gt_id] += 1
            # print(gt_id)

    # Ensure all ground truth IDs are included in the results (set count to 0 if unmatched)
    for gt_id in all_gt_ids:
        if gt_id not in detection_count:
            detection_count[gt_id] = 0
    
    return dict(detection_count)

def evaluate_across_views(aggregated_results, all_gt_ids, total_pred_boxes):
    """
    Evaluates performance across views for a single frame using aggregated detection results.
    
    Args:
        aggregated_results: A dictionary with each GT ID and the number of times it was detected across all views.
        all_gt_ids: A set of all ground truth IDs that should be considered.
        total_pred_boxes: Total number of predicted boxes across all views (used to calculate FP).
    
    Returns:
        A dictionary containing precision, recall, F1-score, TP, FP, FN counts.
    """
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    total_true_positives = 0

    # Calculate TP and FN
    for gt_id in all_gt_ids:
        if aggregated_results.get(gt_id, 0) > 0:
            true_positives += 1  # TP: Detected at least once
            total_true_positives += aggregated_results.get(gt_id, 0)
        else:
            false_negatives += 1  # FN: Not detected at all

    # Calculate FP
    # FP is the total number of predicted boxes minus the true positives
    # This counts predicted boxes that did not match any GT across all views
    false_positives = total_pred_boxes - total_true_positives

    # Precision, Recall, F1-Score Calculation
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def visualize_merged_points(original_points, merged_points, filename="merged_points_plot.png"):
    """
    Visualize the original points and the merged points from DBSCAN, and save the plot as an image file.

    Args:
        all_world_coords: List of world coordinates from different views.
        merged_points: Numpy array of merged points after DBSCAN.
        filename: Filename to save the plot.
    """
    # Convert the list of world coordinates to a numpy array for plotting
    # original_points = np.concatenate(all_world_coords, axis=0)

    # Create a new figure for plotting
    plt.figure(figsize=(8, 6))

    # Plot original points in blue
    plt.scatter(original_points[:, 0], original_points[:, 1], c='blue', label='GT Points')

    # Plot merged points in red
    plt.scatter(merged_points[:, 0], merged_points[:, 1], c='red', label='Pred Points', marker='x')

    # Add labels and titles
    plt.title('Original and Merged Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Add a legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(f"./data/merged_points/{filename}")
    plt.close()