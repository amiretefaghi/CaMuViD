import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import torchvision.transforms.functional as TF
import os
import torch
from torchvision.datasets import VisionDataset
import torch.nn.functional as F
import xml.etree.ElementTree as ET
import re
from PIL import Image, ImageDraw
import shutil
from PIL import ImageFont
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from shapely.geometry import Polygon


def extract_middle_bottem_pt(bbox):
    """Extract the middle bottom points from bounding boxes."""
    points = []
    x_min, y_min, x_max, y_max = bbox
    middle_bottom_x = (x_min + x_max) / 2
    middle_bottom_y = y_max
    point = np.array([middle_bottom_x, middle_bottom_y], dtype=np.float32)
    return point

def generate_projection_matrix(intrinsic_matrix, extrinsic_matrix):
    """
    Generate the camera projection matrix from intrinsic and extrinsic matrices.

    Parameters:
    - intrinsic_matrix: The intrinsic matrix of the camera (3x3 numpy array).
    - extrinsic_matrix: The extrinsic matrix of the camera (3x4 numpy array).

    Returns:
    - The projection matrix (3x4 numpy array).
    """
    # Multiply the intrinsic matrix by the extrinsic matrix to get the projection matrix
    projection_matrix = intrinsic_matrix @ extrinsic_matrix
    # projection_matrix = np.dot(intrinsic_matrix, extrinsic_matrix)
    return projection_matrix

def triangulate_to_ground_plane(point_view1, point_view2, camera_matrix_view1, camera_matrix_view2):
    """
    Perform triangulation for two views given two points and project the result onto the ground plane.

    Parameters:
    - point_view1: The point in the first view (e.g., (x, y) in image coordinates).
    - point_view2: The point in the second view (e.g., (x, y) in image coordinates).
    - camera_matrix_view1: The 3x4 camera (projection) matrix for the first view.
    - camera_matrix_view2: The 3x4 camera (projection) matrix for the second view.

    Returns:
    - The 3D point on the ground plane (x, y, 0) that is the triangulation of the two points.
    """
    # Convert points to homogeneous coordinates
    point_view1_hom = np.array([point_view1[0], point_view1[1]])
    point_view2_hom = np.array([point_view2[0], point_view2[1]])

    # Perform triangulation
    points_4d_hom = cv2.triangulatePoints(camera_matrix_view1, camera_matrix_view2,
                                          point_view1_hom.reshape(2, 1), point_view2_hom.reshape(2, 1))

    # Convert from homogeneous coordinates to 3D
    points_3d = points_4d_hom / points_4d_hom[3]

    # Project the 3D point onto the ground plane by setting z to 0
    ground_point = points_3d[:3]
    ground_point[2] = 0  # Force z-coordinate to 0 to project onto the ground plane
    
    return ground_point

def write_middle_bottom_points(img_id, gt_bboxes, detected_bboxes, gt_filepath, detected_filepath, dataset, camera_ids):
    """
    Write the middle bottom points of ground truth and detected bounding boxes to separate text files.

    Parameters:
    - gt_bboxes: Dict[int, List[List[int]]] - Ground truth bounding boxes with image id as keys.
    - detected_bboxes: Dict[int, List[List[int]]] - Detected bounding boxes with image id as keys.
    - gt_filepath: str - File path for the ground truth text file.
    - detected_filepath: str - File path for the detected bounding boxes text file.
    """

    def extract_and_transform(bboxes, img_id):
        """Extract the middle bottom points from bounding boxes."""
        points = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            middle_bottom_x = (x_min + x_max) / 2
            middle_bottom_y = y_max
            point = np.array([[middle_bottom_x], [middle_bottom_y]], dtype=np.int32)
            world_point = dataset.get_worldcoord_from_imagecoord(point, camera_ids[img_id])
            world_point = dataset.get_worldgrid_from_worldcoord(world_point)
            points.append(world_point)
        return points
    
    def match_points(view1_points, view2_points, distance_threshold):
        """
        Match points between two views using the Hungarian algorithm based on a distance matrix.

        Parameters:
        - view1_points: List of points from view 1.
        - view2_points: List of points from view 2.
        - distance_threshold: The maximum distance between points to be considered a match.

        Returns:
        - matched_pairs: List of tuples, where each tuple contains a point from view 1 and its matching point from view 2.
        """
        num_view1 = len(view1_points)
        num_view2 = len(view2_points)
        view1_indx = []
        view2_indx = []
        # Create a matrix of distances between points
        distance_matrix = np.zeros((num_view1, num_view2))
        
        for i, p1 in enumerate(view1_points):
            for j, p2 in enumerate(view2_points):
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                # If distance is above the threshold, set it to infinity
                distance_matrix[i, j] = distance if distance <= distance_threshold else -10e5
        
        # Apply the Hungarian algorithm (linear sum assignment) to the distance matrix
        row_ind, col_ind = linear_sum_assignment(-distance_matrix)
        
        # Prepare the matched pairs
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            if distance_matrix[i, j] != -10e5:
                # matched_pairs.append((view1_points[i], view2_points[j]))
                matched_pairs.append((i,j))
                # print((i,j))
                view1_indx.append(i)
                view2_indx.append(j)



        return matched_pairs, view1_indx, view2_indx
          
    # def write_to_file(filepath, bboxes):
    #     """Write the middle bottom points to a file."""
    #     with open(filepath, 'a') as file:
    #         for id, boxes in bboxes.items():
    #             for box in boxes:
    #                 middle_bottom_x, middle_bottom_y = extract_middle_bottom_point(box)
    #                 point = np.array([[middle_bottom_x],[middle_bottom_y]],dtype=np.int32)
    #                 p_world_point = dataset.get_worldcoord_from_imagecoord(point, camera_ids[id])
    #                 p_world_point = dataset.get_worldgrid_from_worldcoord(p_world_point)
    #                 file.write(f"{img_id} {p_world_point[0][0]} {p_world_point[1][0]}\n")

    # Function to check if the frame is already written in the file
    def is_frame_written(file_name, frame):
        try:
            with open(file_name, 'r') as file:
                for line in file:
                    if line.startswith(str(frame)):
                        return True
        except FileNotFoundError:
            # File not found means no frame is written
            pass
        return False
    
    def write_to_file(filepath, bbox_dict):
        """Write the middle bottom points to a file."""
        view1_points = extract_and_transform(bbox_dict[1], 1)
        view2_points = extract_and_transform(bbox_dict[2], 2)
        # distance_threshold = 30  # Set this to whatever makes sense for your coordinate system
        # matched_pairs, view1_indx, view2_indx = match_points(view1_points, view2_points, distance_threshold)
        # proj_mat_1 = generate_projection_matrix(dataset.intrinsic_matrices[camera_ids[1]],dataset.extrinsic_matrices[camera_ids[1]])
        # proj_mat_2 = generate_projection_matrix(dataset.intrinsic_matrices[camera_ids[2]],dataset.extrinsic_matrices[camera_ids[2]])
        img_id_1 = str(camera_ids[1]+1) + str(img_id)
        img_id_2 = str(camera_ids[2]+1) + str(img_id)

        if not is_frame_written(filepath, int(img_id_1)):
            with open(filepath, 'a') as file:

                for i in range(len(view1_points)):
                    # if i not in view1_indx:
                        file.write(f"{img_id_1} {view1_points[i][0][0]} {view1_points[i][1][0]}\n")
                # for pair in matched_pairs:
                #     # x = (view1_points[pair[0]][0][0] + view2_points[pair[1]][0][0]) / 2
                #     # y = (view1_points[pair[0]][1][0] + view2_points[pair[1]][1][0]) / 2
                #     pt_1 = extract_middle_bottem_pt(bbox_dict[1][pair[0]])
                #     pt_2 = extract_middle_bottem_pt(bbox_dict[2][pair[1]])
                #     proj_pt = triangulate_to_ground_plane(pt_1,pt_2,proj_mat_1,proj_mat_2)
                #     proj_pt = dataset.get_worldgrid_from_worldcoord(proj_pt[:2])
                #     file.write(f"{img_id} {proj_pt[0][0]} {proj_pt[1][0]}\n")
        if not is_frame_written(filepath, int(img_id_2)):
            with open(filepath, 'a') as file:

                for i in range(len(view2_points)):
                    # if i not in view2_indx:
                        file.write(f"{img_id_2} {view2_points[i][0][0]} {view2_points[i][1][0]}\n")

                # for pair in matched_pairs:
                #     # x = (view1_points[pair[0]][0][0] + view2_points[pair[1]][0][0]) / 2
                #     # y = (view1_points[pair[0]][1][0] + view2_points[pair[1]][1][0]) / 2
                #     pt_1 = extract_middle_bottem_pt(bbox_dict[1][pair[0]])
                #     pt_2 = extract_middle_bottem_pt(bbox_dict[2][pair[1]])
                #     proj_pt = triangulate_to_ground_plane(pt_1,pt_2,proj_mat_1,proj_mat_2)
                #     proj_pt = dataset.get_worldgrid_from_worldcoord(proj_pt[:2])
                #     file.write(f"{img_id} {proj_pt[0][0]} {proj_pt[1][0]}\n")

    
    # Write ground truth and detected bounding boxes' middle bottom points to their respective files
    write_to_file(gt_filepath, gt_bboxes)
    write_to_file(detected_filepath, detected_bboxes)

def ensure_and_clear_directory(directory):
    """ Ensures the directory exists, creates it if it does not, and clears all files and subdirectories within. """
    # Check if the directory exists, create it if it does not
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' was not found and has been created.")
    else:
        # If the directory exists, remove all files and directories within it
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove files or links
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"Directory '{directory}' has been cleared.")

def write_boxes_to_files(gt_boxes, pred_boxes, frame, people_ids, gt_file='ground_truth.txt', pred_file='predictions.txt'):
    # Function to convert box coordinates from [x1, y1, x2, y2] to [bb_left, bb_top, bb_width, bb_height]
    def convert_box(box):
        try:
            x1, y1, x2, y2 = box[:4]
        except:
            print(box)
        bb_left = x1
        bb_top = y1
        bb_width = x2 - x1
        bb_height = y2 - y1
        return bb_left, bb_top, bb_width, bb_height

    # Function to check if the frame is already written in the file
    def is_frame_written(file_name, frame):
        try:
            with open(file_name, 'r') as file:
                for line in file:
                    if line.startswith(str(frame) + ","):
                        return True
        except FileNotFoundError:
            # File not found means no frame is written
            pass
        return False

    # Check and write ground truth boxes if frame not already written
    if not is_frame_written(gt_file, frame):
        if not os.path.isfile(gt_file):
            with open(gt_file, 'w') as file:
                for i, box in enumerate(gt_boxes[0], start=1):
                    bb_left, bb_top, bb_width, bb_height = convert_box(box)
                    line = f"{frame}, {people_ids[i-1]}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, 1.0, -1, -1, -1\n"
                    file.write(line)
        else:

            with open(gt_file, 'a') as file:
                for i, box in enumerate(gt_boxes[0], start=1):
                    bb_left, bb_top, bb_width, bb_height = convert_box(box)
                    line = f"{frame}, {people_ids[i-1]}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, 1.0, -1, -1, -1\n"
                    file.write(line)
    else:
        print(f"Frame {frame} already exists in {gt_file}, skipping write.")

    # Check and write predicted boxes if frame not already written
    if not is_frame_written(pred_file, frame):
        if not os.path.isfile(pred_file):
            with open(pred_file, 'w') as file:
                for i, box in enumerate(pred_boxes, start=1):
                    bb_left, bb_top, bb_width, bb_height = convert_box(box)
                    conf = box[4]  # Confidence is the fifth element in the prediction box array
                    line = f"{frame}, {i}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, {conf}, -1, -1, -1\n"
                    file.write(line)
        else:        
            with open(pred_file, 'a') as file:
                for i, box in enumerate(pred_boxes, start=1):
                    bb_left, bb_top, bb_width, bb_height = convert_box(box)
                    conf = box[4]  # Confidence is the fifth element in the prediction box array
                    line = f"{frame}, {i}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, {conf}, -1, -1, -1\n"
                    file.write(line)
    else:
        print(f"Frame {frame} already exists in {pred_file}, skipping write.")


def plot_world_points_and_save(gt_bboxes, detected_bboxes, dataset, camera_ids, save_path):
    """
    Plot the world points of ground truth and detected bounding boxes on the same plane and save the plot to a file.
    Ensure that the legend does not repeat labels.
    """
    fig, ax = plt.subplots()
    colors = {'gt': 'blue', 'detected': 'red'}

    # Create a list to keep track of which labels have been used
    used_labels = set()

    # def extract_and_transform(bbox, img_id):
    #     middle_bottom_x, middle_bottom_y = (bbox[0] + bbox[2]) / 2, bbox[3]
    #     point = np.array([[middle_bottom_x], [middle_bottom_y]], dtype=np.int32)
    #     world_point = dataset.get_worldcoord_from_imagecoord(point, camera_ids[img_id])
    #     return dataset.get_worldgrid_from_worldcoord(world_point)
    def extract_and_transform(bboxes, img_id):
        """Extract the middle bottom points from bounding boxes."""
        points = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            middle_bottom_x = (x_min + x_max) / 2
            middle_bottom_y = y_max
            point = np.array([[middle_bottom_x], [middle_bottom_y]], dtype=np.int32)
            world_point = dataset.get_worldcoord_from_imagecoord(point, camera_ids[img_id])
            world_point = dataset.get_worldgrid_from_worldcoord(world_point)
            points.append(world_point)
        return points
    
    def match_points(view1_points, view2_points, distance_threshold):
        """
        Match points between two views using the Hungarian algorithm based on a distance matrix.

        Parameters:
        - view1_points: List of points from view 1.
        - view2_points: List of points from view 2.
        - distance_threshold: The maximum distance between points to be considered a match.

        Returns:
        - matched_pairs: List of tuples, where each tuple contains a point from view 1 and its matching point from view 2.
        """
        num_view1 = len(view1_points)
        num_view2 = len(view2_points)
        view1_indx = []
        view2_indx = []
        # Create a matrix of distances between points
        distance_matrix = np.zeros((num_view1, num_view2))
        
        for i, p1 in enumerate(view1_points):
            for j, p2 in enumerate(view2_points):
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                # If distance is above the threshold, set it to infinity
                distance_matrix[i, j] = distance if distance <= distance_threshold else -10e5
        
        # Apply the Hungarian algorithm (linear sum assignment) to the distance matrix
        row_ind, col_ind = linear_sum_assignment(-distance_matrix)
        
        # Prepare the matched pairs
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            if distance_matrix[i, j] != -10e5:
                # matched_pairs.append((view1_points[i], view2_points[j]))
                matched_pairs.append((i,j))
                # print((i,j))
                view1_indx.append(i)
                view2_indx.append(j)



        return matched_pairs, view1_indx, view2_indx

    def common_points(bbox_dict):
        """Write the middle bottom points to a file."""
        view1_points = extract_and_transform(bbox_dict[1], 1)
        view2_points = extract_and_transform(bbox_dict[2], 2)
        distance_threshold = 10  # Set this to whatever makes sense for your coordinate system
        matched_pairs, view1_indx, view2_indx = match_points(view1_points, view2_points, distance_threshold)
        proj_mat_1 = generate_projection_matrix(dataset.intrinsic_matrices[camera_ids[1]],dataset.extrinsic_matrices[camera_ids[1]])
        proj_mat_2 = generate_projection_matrix(dataset.intrinsic_matrices[camera_ids[2]],dataset.extrinsic_matrices[camera_ids[2]])
        det_points = []
        for i in range(len(view1_points)):
            if i not in view1_indx:
                det_points.append([view1_points[i][0][0], view1_points[i][1][0]])

        for i in range(len(view2_points)):
            if i not in view2_indx:
                det_points.append([view2_points[i][0][0], view2_points[i][1][0]])

        for pair in matched_pairs:
            x = (view1_points[pair[0]][0][0] + view2_points[pair[1]][0][0]) / 2
            y = (view1_points[pair[0]][1][0] + view2_points[pair[1]][1][0]) / 2
            det_points.append([x, y])
            # pt_1 = extract_middle_bottem_pt(bbox_dict[1][pair[0]])
            # pt_2 = extract_middle_bottem_pt(bbox_dict[2][pair[1]])
            # proj_pt = triangulate_to_ground_plane(pt_1,pt_2,proj_mat_1,proj_mat_2)
            # proj_pt = dataset.get_worldgrid_from_worldcoord(proj_pt[:2])
            # det_points.append([proj_pt[0][0], proj_pt[1][0]])
        return det_points

    det_points = common_points(detected_bboxes)
    gt_points = common_points(gt_bboxes)
    # Plot GT boxes
    # for img_id, boxes in gt_bboxes.items():
    #     for box in boxes:
    #         world_point = extract_and_transform(box, img_id)
    #         label = 'GT' if 'GT' not in used_labels else ""
    #         ax.scatter(world_point[0][0], world_point[1][0], color=colors['gt'], label=label)
    #         used_labels.add('GT')

    for point in gt_points:
        label = 'GT' if 'GT' not in used_labels else ""
        ax.scatter(point[0], point[1], color=colors['gt'], label=label)
        used_labels.add('GT')

    # Plot detected boxes
    # for img_id, boxes in detected_bboxes.items():
    #     for box in boxes:
    #         world_point = extract_and_transform(box, img_id)
    #         label = 'Detected' if 'Detected' not in used_labels else ""
    #         ax.scatter(world_point[0][0], world_point[1][0], color=colors['detected'], label=label)
    #         used_labels.add('Detected')

    for point in det_points:
        label = 'Detected' if 'Detected' not in used_labels else ""
        ax.scatter(point[0], point[1], color=colors['detected'], label=label)
        used_labels.add('Detected')

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.set_title('World Points of GT and Detected Bounding Boxes')

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

class ResizeKeepRatio:
    def __init__(self, img_scale, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def __call__(self, image, target=None):
        if self.keep_ratio:
            # w, h = image.size
            # min_side, max_side = self.img_scale
            # scale_factor = min(min_side / min(h, w), max_side / max(h, w))

            w, h = image.size
            h_, w_ = self.img_scale
            scale_factor = min(h_/h, w_/w)
            # new_w, new_h = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
            new_h, new_w = int(h * scale_factor + 0.5), int(w * scale_factor + 0.5)

            image = TF.resize(image, [new_h, new_w])

            if target is not None and 'boxes' in target:
                # Adjust bounding boxes if target is provided
                # Assume target['boxes'] is in [x_min, y_min, x_max, y_max] format
                boxes = target['boxes']
                # Rescale bounding boxes according to the image's scale factor
                boxes = boxes * torch.tensor([scale_factor, scale_factor, scale_factor, scale_factor], dtype=torch.float32)
                target['boxes'] = boxes
        else:
            image = TF.resize(image, self.img_scale)
        return image

class RandomFlip:
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio
    
    def __call__(self, image, target=None):
        if torch.rand(1) < self.flip_ratio:
            image = F.hflip(image)
            if target is not None:
                w, _ = image.size
                target['boxes'][:, [0, 2]] = w - target['boxes'][:, [2, 0]]
        return image

class Normalize:
    def __init__(self, mean, std, to_rgb=True):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
    
    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image

class PadToDivisor:
    def __init__(self, size_divisor):
        self.size_divisor = size_divisor
    
    def __call__(self, image, target=None):
        h, w = image.shape[-2:]
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor
        image = F.pad(image, (0, 0, pad_w, pad_h))
        return image


class PadToSizeDivisibleBy:
    """Pad image to ensure its dimensions are divisible by a given number."""
    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, img):
        width, height = img.size()[1:]
        pad_height = (self.divisor - height % self.divisor) % self.divisor
        pad_width = (self.divisor - width % self.divisor) % self.divisor
        
        # Only pad if necessary
        if pad_height > 0 or pad_width > 0:
            img = TF.pad(img, padding=(0, 0, pad_height, pad_width), padding_mode='constant', fill=0)
        
        return img

class PadToSizeDivisibleBy_BATCH:
    """Pad image to ensure its dimensions are divisible by a given number."""
    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, img):

        width, height = img.size()[2:]
        pad_height = (self.divisor - height % self.divisor) % self.divisor
        pad_width = (self.divisor - width % self.divisor) % self.divisor
        
        # Only pad if necessary
        if pad_height > 0 or pad_width > 0:
            img = TF.pad(img, padding=(0, 0, pad_height, pad_width), padding_mode='constant', fill=0)
        
        return img
    
def get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
    projection_matrices = {}
    for cam in range(7):
        worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

        worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
        imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
        # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
        # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
        permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        pass
    return projection_matrices    

class Wildtrack(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # WILDTRACK has ij-indexing: H*W=480*1440, so x should be \in [0,480), y \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation,
        self.intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                            'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
        self.extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                            'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']
        self.__name__ = 'Wildtrack'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 7, 2000
        # x,y actually means i,j in Wildtrack, which correspond to h,w
        self.indexing = 'ij'
        # i,j for world map indexing
        self.worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 480

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord
        grid_x = (coord_x + 300) / 2.5
        grid_y = (coord_y + 900) / 2.5
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             self.intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           self.extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix
    

    def get_worldcoord_from_imagecoord(self,image_coord, cam_id):
        project_mat = self.intrinsic_matrices[cam_id] @ self.extrinsic_matrices[cam_id]
        project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
        image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
        world_coord = project_mat @ image_coord
        world_coord = world_coord[:2, :] / world_coord[2, :]
        return world_coord


    def get_imagecoord_from_worldcoord(self, world_coord, cam_id):
        project_mat = self.intrinsic_matrices[cam_id] @ self.extrinsic_matrices[cam_id]
        project_mat = np.delete(project_mat, 2, 1)
        world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
        image_coord = project_mat @ world_coord
        image_coord = image_coord[:2, :] / image_coord[2, :]
        return image_coord

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam
    
class MultiviewX(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # MultiviewX has xy-indexing: H*W=640*1000, thus x is \in [0,1000), y \in [0,640)
        # MultiviewX has consistent unit: meter (m) for calibration & pos annotation
        self.intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                            'intr_Camera5.xml', 'intr_Camera6.xml']
        self.extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                            'extr_Camera5.xml', 'extr_Camera6.xml']
        self.__name__ = 'MultiviewX'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [640, 1000]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 6, 400
        # x,y correspond to w,h
        self.indexing = 'xy'
        # convert x,y to i,j, then use i,j for world map indexing
        self.worldgrid2worldcoord_mat = np.array([[0, 0.025, 0], [0.025, 0, 0], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])



    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 1000

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord
        grid_x = coord_x * 40
        grid_y = coord_y * 40
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = grid_x / 40
        coord_y = grid_y / 40
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                      self.intrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                      self.extrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix
    

    def get_worldcoord_from_imagecoord(self,image_coord, cam_id):
        project_mat = self.intrinsic_matrices[cam_id] @ self.extrinsic_matrices[cam_id]
        project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
        image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
        world_coord = project_mat @ image_coord
        world_coord = world_coord[:2, :] / world_coord[2, :]
        return world_coord


    def get_imagecoord_from_worldcoord(self, world_coord, cam_id):
        project_mat = self.intrinsic_matrices[cam_id] @ self.extrinsic_matrices[cam_id]
        project_mat = np.delete(project_mat, 2, 1)
        world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
        image_coord = project_mat @ world_coord
        image_coord = image_coord[:2, :] / image_coord[2, :]
        return image_coord

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam

# Custom function to move img_metas to GPU
def move_to_device(meta, device):
    if isinstance(meta, dict):
        return {k: move_to_device(v, device) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [move_to_device(v, device) for v in meta]
    elif isinstance(meta, tuple):
        return tuple(move_to_device(v, device) for v in meta)
    elif isinstance(meta, torch.Tensor):
        return meta.to(device).squeeze(0)
    else:
        return meta

def create_plane(point1, point2, point3):
    # Convert the points to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    # Calculate the direction vectors
    vector1 = p2 - p1
    vector2 = p3 - p1

    # Calculate the cross product to obtain the normal vector
    normal_vector = np.cross(vector1, vector2)

    # Extract the coefficients A, B, and C from the normal vector
    A, B, C = normal_vector

    # Calculate the value of D using any of the points
    D = -np.dot(normal_vector, p1)

    # Return the coefficients A, B, C, and D as a tuple
    return A, B, C, D

def create_plane_2d(coordinates, fixed_z):
    # Convert the coordinates to numpy arrays
    coords = np.array(coordinates)

    # Add the fixed z-coordinate to all points
    points_3d = np.column_stack((coords, np.full(len(coords), fixed_z)))

    # Calculate the coefficients A, B, C, and D using the 3D coordinates
    A, B, C, D = create_plane(points_3d[0], points_3d[1], points_3d[2])

    # Return the coefficients A, B, C, and D as a tuple
    return A, B, C, D

def is_point_within_grid(point, grid_points):
    # Find the minimum and maximum x and y values of the grid points
    x_values = [p[0] for p in grid_points]
    y_values = [p[1] for p in grid_points]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    # Check if the point is within the grid boundaries
    x, y = point
    return min_x <= x <= max_x and min_y <= y <= max_y

def process_targets(bboxes, labels, masks, scale_factor, device, evaluation=False):
    """Prepare targets and labels for training."""
    # Initialize lists to store processed targets and labels
    scaled_bboxes = []
    adjusted_labels = []
    c_masks = []

    # for t in target_list:
    #     # Calculate scaled bounding box and adjust category id, then convert to tensor
    #     if torch.is_tensor(t['bbox']):
    #         scaled_bbox = t['bbox'][0] * scale_factor
    #         category_id = t['category_id'][0] - 1  # Adjust category ID if necessary
    #     else:
    #         scaled_bbox = torch.tensor(t['bbox']) * scale_factor
    #         category_id = torch.tensor(t['category_id']) - 1  # Adjust category ID if necessary

    #     # Append tensors to the lists
    #     targets.append(scaled_bbox.to(device))
    #     labels.append(torch.tensor(category_id, device=device))  # Store label as tensor within a list
    for i in range(len(bboxes)):
        if torch.is_tensor(bboxes[i]):
            if not evaluation:
                scaled_bboxes.append(bboxes[i].to(device=device)*scale_factor)
            else:
                scaled_bboxes.append(bboxes[i].to(device=device))
            adjusted_labels.append(labels[i].squeeze(-1).to(device=device) - 1)
        else:
            if not evaluation:
                scaled_bboxes.append(torch.tensor(bboxes[i]).unsqueeze(0).to(device=device)*scale_factor)
            else:
                scaled_bboxes.append(torch.tensor(bboxes[i]).unsqueeze(0).to(device=device))

            adjusted_labels.append(torch.tensor(labels[i]).squeeze(-1).to(device=device) - 1)
        # c_masks.append(masks[i].to(device=device))
    # Convert the list of target tensors to a single tensor

    # targets_tensor = [torch.stack(targets).to(device=device)]
    # labels_tensor = [torch.stack(labels)]

    # Return the targets tensor and the tuple of label tensors
    return scaled_bboxes, adjusted_labels, c_masks


def get_projection_matrices(target_lists, dataset, camera_ids, device):
    """Compute projection matrices for the given camera views."""
    anchor_camera, pair_camera = target_lists[0][0]['camera'][0], target_lists[1][0]['camera'][0]
    anchor_ids, pair_ids = camera_ids[anchor_camera], camera_ids[pair_camera]
    anchor_proj = generate_projection_matrix(dataset.intrinsic_matrices[anchor_ids], dataset.extrinsic_matrices[anchor_ids])
    pair_proj = generate_projection_matrix(dataset.intrinsic_matrices[pair_ids], dataset.extrinsic_matrices[pair_ids])
    return torch.tensor(anchor_proj, device=device), torch.tensor(pair_proj, device=device)

def log_training(writer, loss_dicts, epoch, batch_index, total_batches, single_head=False):
    """Log training losses using TensorBoard."""
    for i,loss_dict in enumerate(loss_dicts.values(),1):
        if not single_head:
            # loss_keys = set(loss_dict.keys()).union(set(loss_dict_2.keys()))
            # for key in loss_keys:
            for key in loss_dict.keys():
                # if key in loss_dict_1:
                if isinstance(loss_dict[key], list):
                    for j, loss in enumerate(loss_dict[key]):
                        writer.add_scalar(f'Loss/{key}_{i}_{j}', loss.item(), epoch * total_batches + batch_index)
                else:
                    writer.add_scalar(f'Loss/{key}_{i}', loss_dict[key].item(), epoch * total_batches + batch_index)
                    # writer.add_scalar(f'Loss/{key}_1', loss_dict_1[key].item(), epoch * total_batches + batch_index)
                # if key in loss_dict_2:
                #     if isinstance(loss_dict_2[key], list):
                #         for i, loss in enumerate(loss_dict_2[key]):
                #             writer.add_scalar(f'Loss/{key}_2_{i}', loss.item(), epoch * total_batches + batch_index)
                #     else:
                #         writer.add_scalar(f'Loss/{key}_2', loss_dict_2[key].item(), epoch * total_batches + batch_index)
        else:
            for key in loss_dict_1.keys():
                if isinstance(loss_dict_1[key], list):
                    for i, loss in enumerate(loss_dict_1[key]):
                        writer.add_scalar(f'Loss/{key}_1_{i}', loss.item(), epoch * total_batches + batch_index)
                else:
                    writer.add_scalar(f'Loss/{key}_1', loss_dict_1[key].item(), epoch * total_batches + batch_index)

def draw_bboxes_on_image(results, dataset, batch_index, scale_factor, targets, single_head = False, evaluation=False, fov_filtering=None, camera_ids=None, p_idxs=None,people_ids=None):
    imgs = []
    view_names = []
    anchor_image_info = dataset.images[batch_index]
    anchorview = anchor_image_info['file_name'].split('/')[0]
    # frame = anchor_image_info['file_name'].split('/')[1].split('.')[0]
    anchor_image_path = os.path.join(dataset.root_dir, anchor_image_info['file_name'])
    anchor_image = Image.open(anchor_image_path).convert("RGB")
    imgs.append(anchor_image)
    view_names.append(anchorview)  # Store anchor view name

    if evaluation:
        for p_idx in p_idxs:
            positive_pair = dataset.positive_pairs[batch_index][p_idx]
            pairview = positive_pair['file_name'].split('/')[0]
            pair_image_path_p = os.path.join(dataset.root_dir, positive_pair['file_name'])
            positive_pair_image = Image.open(pair_image_path_p).convert("RGB")
            imgs.append(positive_pair_image)
            view_names.append(pairview)  # Store positive pair view name
    else:
        for p_idx in dataset.p_idx:
            positive_pair = dataset.positive_pairs[batch_index][p_idx]
            pairview = positive_pair['file_name'].split('/')[0]
            pair_image_path_p = os.path.join(dataset.root_dir, positive_pair['file_name'])
            positive_pair_image = Image.open(pair_image_path_p).convert("RGB")
            imgs.append(positive_pair_image)
            view_names.append(pairview)  # Store positive pair view name

    canvases = []
    all_det_bboxes = dict()

    # # Load a font with the desired size. You can specify the font path or use the default PIL font.
    # font_size = 20  # Adjust the size to your preference
    # font = ImageFont.truetype("arial.ttf", font_size)  # You can replace "arial.ttf" with the path to any font you want to use
    width = 4
    for i, image in enumerate(imgs):

        # Draw bounding boxes on the anchor image
        draw_anchor = ImageDraw.Draw(image)
        det_bboxes = []

        result_key = f'results_{i+1}'
        detection_key = f'detects_{i+1}'
        target = targets[i]
        j = 0
        for box in results[result_key][0][0]:  # Assuming results[result_key][0] contains bbox for the image
            # if box[-1] >= 0.85: # multiviewX
            if box[-1] >= 0.85: # Wildtrack
                # Define the label (e.g., confidence score or class name)
                  # Example: Confidence score
                # Calculate bounding box coordinates
                x1, y1, x2, y2 = box[0] * (1 / scale_factor), box[1] * (1 / scale_factor), box[2] * (1 / scale_factor), box[3] * (1 / scale_factor)

                if evaluation:
                    if fov_filtering is not None:
                        if fov_filtering.convert_imgcoord_to_worldgrid(box[:4] * (1 / scale_factor), camera_ids[i]):
                            label = f"{j}"
                            draw_anchor.rectangle(((x1, y1), (x2, y2)), outline="red", width=width)
                            det_bboxes.append([x1, y1, x2, y2, box[-1]])
                            # Draw the label text just above the bounding box
                            text_position = (x1, y1)  # Position the text just above the top-left corner of the bbox
                            # draw_anchor.text(text_position, label, fill="white")  # Use white text color for visibility
                            j += 1
                        else:
                            draw_anchor.rectangle(((x1, y1), (x2, y2)), outline="blue", width=width)
                    else:
                        label = f"{j}"
                        draw_anchor.rectangle(((x1, y1), (x2, y2)), outline="red", width=width)
                        det_bboxes.append([x1, y1, x2, y2, box[-1]])
                        text_position = (x1, y1)  # Position the text just above the top-left corner of the bbox
                        # draw_anchor.text(text_position, label, fill="white")  # Use white text color for visibility
                        j += 1
                else:
                    draw_anchor.rectangle(((x1, y1), (x2, y2)), outline="red", width=width)
                    det_bboxes.append([x1, y1, x2, y2, box[-1]])

        for j, box in enumerate(target[0]):  # Assuming targets[i][0] contains bbox for the image
            if box.device == 'cuda':
                box = box.cpu()
            if evaluation:
                label = f"{people_ids[i][j]}"
                text_position = (box[2], box[1])  # Position the text just above the top-left corner of the bbox
                # draw_anchor.text(text_position, label, fill="gray")  # Use white text color for visibility
                draw_anchor.rectangle(((box[0], box[1]), (box[2], box[3])), outline="green", width=width)
            else:
                
                draw_anchor.rectangle(((box[0] * (1 / scale_factor), box[1] * (1 / scale_factor)), 
                                       (box[2] * (1 / scale_factor), box[3] * (1 / scale_factor))), outline="green", width=width)

        # Convert PIL image back to cv2 image
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not evaluation:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        canvases.append(image_cv)
        all_det_bboxes[detection_key] = det_bboxes

    # Combine all images into one canvas
    max_height = max(canvas.shape[0] for canvas in canvases)
    total_width = sum(canvas.shape[1] for canvas in canvases)

    # Assuming you have 7 views (canvases)
    n_views = len(canvases)
    n_columns = 2
    n_rows = (n_views + n_columns - 1) // n_columns  # This will calculate the required number of rows

    # Assuming all canvases have the same dimensions
    canvas_height = max_height + 50  # Adjusted for text space
    canvas_width = max(canvas.shape[1] for canvas in canvases)

    # Calculate the total canvas size needed
    total_height = n_rows * (max_height + 50)
    total_width = n_columns * canvas_width

    # Create a blank combined canvas with the correct size
    combined_canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    for i, canvas in enumerate(canvases):
        row = i // n_columns
        col = i % n_columns

        x_offset = col * canvas_width
        y_offset = row * (max_height + 50)

        # Paste the canvas at the computed row and column position
        combined_canvas[y_offset:y_offset + canvas.shape[0], x_offset:x_offset + canvas.shape[1]] = canvas

        # Add text below the image
        cv2.putText(combined_canvas, view_names[i], (x_offset + 10, y_offset + max_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # # Combine all images into one canvas
    # max_height = max(canvas.shape[0] for canvas in canvases)
    # total_width = sum(canvas.shape[1] for canvas in canvases)

    # canvas_height = max_height + 50  # Adding extra space for text
    # combined_canvas = np.ones((canvas_height, total_width, 3), dtype=np.uint8) * 255

    # current_width = 0
    # for i, canvas in enumerate(canvases):

    #     combined_canvas[:canvas.shape[0], current_width:current_width + canvas.shape[1]] = canvas
    #     cv2.putText(combined_canvas, view_names[i], (current_width + 10, max_height + 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    #     current_width += canvas.shape[1]

    if evaluation:
        return combined_canvas, all_det_bboxes
    else:
        return combined_canvas

def log_training_img(writer, model, imgs, img_metas, dataset, epoch, batch_index, total_batches, targets, single_head=False,scale_factor=None, distributed=True):
    """Log training images using TensorBoard."""
    ## get the bounding boxes
    for i in range(len(imgs)):
        if not single_head:
            if distributed:
                results = model.module.simple_test(imgs=imgs, img_metas = img_metas, rescale=False)
            else:
                results = model.simple_test(imgs=imgs, img_metas = img_metas, rescale=False)
            # scale_factor = img_metas[0]['scale_factor'][0]
            canvas = draw_bboxes_on_image(results, dataset, batch_index, scale_factor, targets, single_head=single_head)
        else: 
            results_1 = model.simple_test(img1 = img1,img2 = img2, img_metas = img_metas, rescale=False)
            # scale_factor = img_metas[0]['scale_factor'][:2]
            canvas = draw_bboxes_on_image(results_1, dataset, batch_index, scale_factor, targets_1, targets_2, single_head=single_head)
        # Log the images
        writer.add_image(f'Image/concatenated', torch.from_numpy(canvas.transpose(2, 0, 1)), epoch*total_batches + batch_index)
        # writer.add_image(f'Image/img2', img2, epoch * total_batches + batch_index)

def print_batch_loss(epoch, num_epochs, batch_index, total_batches, running_loss, loss_dicts):
    """Print loss information every few batches."""
    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index}/{total_batches}], Total Loss: {running_loss/batch_index:.4f}")

    # specific_keys = ['loss_bbox', 'loss_cls', 'loss_iou', 'interm_loss_iou', 'acc', 's0.loss_cls', 's0.loss_bbox', 's1.loss_cls', 's1.loss_bbox', 's2.loss_cls', 's2.loss_bbox']
    # for loss_dict in loss_dicts.values():
    #     for key in specific_keys:
    #         # Check if the key exists in the dictionaries, print 'N/A' if it doesn't
    #         loss = loss_dict.get(key, torch.tensor(float('nan')))  # Use nan for missing values
    #         # loss_2 = loss_dict_2.get(key, torch.tensor(float('nan')))  # Use nan for missing values
    #         # Format the output to handle both present and missing (nan) cases
    #         loss_str = f"{loss.item():.4f}" if not torch.isnan(loss) else 'N/A'
    #         # loss_2_str = f"{loss_2.item():.4f}" if not torch.isnan(loss_2) else 'N/A'

    #         if loss_str == 'N/A':
    #             continue
    #         # Print the formatted string for each specific key
    #         print(f"{key} : {loss_str}")
    #     # print(f"{key} 1: {loss_1_str}")


def log_epoch_end(epoch, num_epochs, running_loss, total_batches, scheduler):
    """Log end of epoch details."""
    epoch_loss = running_loss / total_batches
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr}")

def calculate_total_loss(loss_dicts, head='rcnn', single_head=False):
    total_loss = 0
    for loss_dict in loss_dicts.values():
        if not head=='dino':
            rpn_cls_loss = sum(loss_dict['loss_rpn_cls'])
            rpn_bbox_loss = sum(loss_dict['loss_rpn_bbox'])
            # if not single_head:
            #     rpn_cls_loss_2 = sum(loss_dicts[1]['loss_rpn_cls'])
            #     rpn_bbox_loss_2 = sum(loss_dicts[1]['loss_rpn_bbox'])
            if head == 'rcnn':
                # if not single_head:
                #     roi_cls_loss_2 = loss_dicts[1]['loss_cls']
                #     roi_bbox_loss_2 = loss_dicts[1]['loss_bbox']
                roi_cls_loss = loss_dict['loss_cls']
                roi_bbox_loss = loss_dict['loss_bbox']

                total_loss += rpn_cls_loss + rpn_bbox_loss + roi_cls_loss + roi_bbox_loss

                # if single_head:
                #     total_loss = rpn_cls_loss_1 + rpn_bbox_loss_1 + roi_cls_loss_1 + roi_bbox_loss_1
                # else:
                #     total_loss = rpn_cls_loss_1 + rpn_bbox_loss_1 + roi_cls_loss_1 + roi_bbox_loss_1 + rpn_cls_loss_2 + rpn_bbox_loss_2 + roi_cls_loss_2 + roi_bbox_loss_2
            elif head == 'cascade':
                # if not single_head:
                #     roi_cls_loss_2 = loss_dicts[1]['s0.loss_cls'] + loss_dicts[1]['s1.loss_cls'] + loss_dicts[1]['s2.loss_cls']
                #     roi_bbox_loss_2 = loss_dicts[1]['s0.loss_bbox'] + loss_dicts[1]['s1.loss_bbox'] + loss_dicts[1]['s2.loss_bbox']
                roi_cls_loss = loss_dict['s0.loss_cls'] + loss_dict['s1.loss_cls'] + loss_dict['s2.loss_cls']
                roi_bbox_loss = loss_dict['s0.loss_bbox'] + loss_dict['s1.loss_bbox'] + loss_dict['s2.loss_bbox']
                inverse_loss = loss_dict['inverse_loss']

                total_loss += rpn_cls_loss + rpn_bbox_loss + roi_cls_loss + roi_bbox_loss + inverse_loss
                
                # if single_head:
                #     total_loss = rpn_cls_loss_1 + rpn_bbox_loss_1 + roi_cls_loss_1 + roi_bbox_loss_1
                # else:
                #     total_loss = rpn_cls_loss_1 + rpn_bbox_loss_1 + roi_cls_loss_1 + roi_bbox_loss_1 + rpn_cls_loss_2 + rpn_bbox_loss_2 + roi_cls_loss_2 + roi_bbox_loss_2
        else:
            total_loss = 0.0

            if not single_head:
                # Iterate over each dictionary passed to the function
                for loss_dict in loss_dicts:
                    for key, value in loss_dict.items():
                        # Check if 'loss' is in the key name
                        if 'loss' in key:
                            if isinstance(value, list):
                                # If the value is a list, sum its elements
                                total_loss += sum(value)
                            else:
                                # Otherwise, directly add the value
                                total_loss += value     
            else:
                # Iterate over each dictionary passed to the function
                for key, value in loss_dicts[0].items():
                    # Check if 'loss' is in the key name
                    if 'loss' in key:
                        if isinstance(value, list):
                            # If the value is a list, sum its elements
                            total_loss += sum(value)
                        else:
                            # Otherwise, directly add the value
                            total_loss += value       

    return total_loss

def convert_img_meta(img_meta):
    def tensor_to_tuple(tensor_value):
        return tuple(tensor_value.tolist()) if hasattr(tensor_value, 'tolist') else tensor_value
    if img_meta['scale_factor'].dim() == 1:
        scale_factor = (int(img_meta['scale_factor'][0]),int(img_meta['scale_factor'][0]))
    else:
        scale_factor = tuple(img_meta['scale_factor'][0])
    converted_img_meta = {
        'filename': img_meta['filename'],
        'ori_filename': img_meta['ori_filename'],
        'ori_shape': (int(img_meta['ori_shape'][0][0]), int(img_meta['ori_shape'][1][0]), int(img_meta['ori_shape'][2][0])),
        'img_shape': (int(img_meta['img_shape'][0][0]), int(img_meta['img_shape'][1][0]), int(img_meta['img_shape'][2][0])),
        'pad_shape': (int(img_meta['pad_shape'][0][0]), int(img_meta['pad_shape'][1][0]), int(img_meta['pad_shape'][2][0])),
        'scale_factor': scale_factor,
        'flip': bool(img_meta['flip']),
        'img_norm_cfg': {
            'mean': tuple(img_meta['img_norm_cfg']['mean'][0]),
            'std': tuple(img_meta['img_norm_cfg']['std'][0]),
            'to_rgb': bool(img_meta['img_norm_cfg']['to_rgb'])
        },
        'batch_input_shape': tuple(img_meta['batch_input_shape'])
    }
    
    return converted_img_meta

class FOV_Filtering:

    def __init__(self,dataset):
        self.dataset = dataset
        self.xi = np.arange(0, self.dataset.worldgrid_shape[0], 40)
        self.yi = np.arange(0, self.dataset.worldgrid_shape[1], 40)
        self.world_grid = np.stack(np.meshgrid(self.xi, self.yi, indexing='ij')).reshape([2, -1])

    def is_point_on_grid(self, point):
        x, y = point
        return 40 < x[0] and x[0] < (self.dataset.worldgrid_shape[0])  and 95 < y[0] and y[0] < (self.dataset.worldgrid_shape[1] - 150) # Wildtrack
        # return 0 < x[0] and x[0] < (self.dataset.worldgrid_shape[1])  and 0 < y[0] and y[0] < (self.dataset.worldgrid_shape[0]) #multiviewx


    def convert_imgcoord_to_worldgrid(self, bbox,camera_id):

        x_min, y_min, x_max, y_max = bbox
        middle_bottom_x = (x_min + x_max) / 2
        middle_bottom_y = y_max
        point = np.array([[middle_bottom_x], [middle_bottom_y]], dtype=np.int32)
        world_coord = self.dataset.get_worldcoord_from_imagecoord(point,camera_id)
        world_grid_pt = self.dataset.get_worldgrid_from_worldcoord(world_coord)
        return self.is_point_on_grid(world_grid_pt)

def create_seqinfo(name, gt_path, im_dir="img1", frame_rate=5, seq_length=40, im_width=1920, im_height=1080, im_ext=".jpg"):
    seqinfo_content = f"""
[Sequence]
name={name}
imDir={im_dir}
frameRate={frame_rate}
seqLength={seq_length}
imWidth={im_width}
imHeight={im_height}
imExt={im_ext}
"""
    with open(os.path.join(gt_path,"seqinfo.ini"), "w") as f:
        f.write(seqinfo_content.strip())
        
def cross_category_nms(boxes, scores, categories, iou_threshold=0.5):
    """
    Apply NMS among detections from different categories.

    Args:
        boxes (Tensor[N, 4]): Bounding boxes.
        scores (Tensor[N]): Scores for each bounding box.
        categories (Tensor[N]): Category labels for each bounding box.
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        keep (Tensor): Indices of the kept boxes.
    """
    
    # Step 1: Calculate the IoU matrix
    iou_matrix = box_iou(boxes, boxes)

    # Step 2: Create the category matrix
    category_matrix = (categories.unsqueeze(0) == categories.unsqueeze(1)).float()

    # Step 3: Create the weight matrix
    weight_matrix = 1 - category_matrix

    # Step 4: Apply the weight matrix to the IoU matrix
    weighted_iou_matrix = iou_matrix * weight_matrix

    # Step 5: Perform NMS considering weighted_iou_matrix
    keep = []
    idxs = scores.argsort(descending=True)

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current.item())
        if len(idxs) == 1:
            break
        idxs = idxs[1:]
        ious = weighted_iou_matrix[current, idxs]
        idxs = idxs[ious <= iou_threshold]

    return torch.tensor(keep)


def apply_nms_to_detections(detections, idxs, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to each frame of detections.

    :param detections: Nested dictionary with camera names as keys and another dictionary as values.
                       The inner dictionary has frame numbers as keys and lists of bounding boxes with scores as values.
    :param iou_threshold: IoU threshold for NMS.
    :return: Updated dictionary with suppressed bounding boxes.
    """
    
    suppressed_detections = {}
    
    for camera, frames in detections.items():
        suppressed_detections[camera] = {}
        for frame, boxes in frames.items():
            if len(boxes) == 0:
                suppressed_detections[camera][frame] = []
                continue
            
            # Convert boxes to tensors
            boxes_tensor = torch.tensor([box[:4] for box in boxes], dtype=torch.float32)
            scores_tensor = torch.tensor([box[4] for box in boxes], dtype=torch.float32)
            idxs_tensor = torch.tensor(idxs[camera][frame])
            # Apply NMS
            selected_indices = ops.nms(
                boxes=boxes_tensor,
                scores=scores_tensor,
                # categories=idxs_tensor,
                iou_threshold=iou_threshold
            )
            
            # Get the selected boxes
            suppressed_boxes = [boxes[i] for i in selected_indices]
            suppressed_detections[camera][frame] = suppressed_boxes
    
    return suppressed_detections

def category_aware_nms(boxes, scores, categories, iou_threshold):
    """
    Perform category-aware non-maximum suppression (NMS) on the bounding boxes.

    Parameters:
    boxes (torch.Tensor): Tensor of bounding boxes in the format [x1, y1, x2, y2].
    scores (torch.Tensor): Tensor of confidence scores for each bounding box.
    categories (torch.Tensor): Tensor of category labels for each bounding box.
    iou_threshold (float): IoU threshold for suppression.

    Returns:
    torch.Tensor: Indices of the bounding boxes to keep.
    """

    # If no boxes, return empty list
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)

    # Convert to float if not already
    boxes = boxes.float()

    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort by score
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = []

    while order.size(0) > 0:
        i = order[0]
        keep.append(i.item())

        # Compute IoU of the kept box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Indices of boxes with IoU less than the threshold or different category
        inds = torch.where((iou <= iou_threshold) | (categories[order[1:]] != categories[i]))[0]

        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long)

def visualize_feature_map(feature_map, filename=None, colormap=cv2.COLORMAP_JET, display=False):
    """
    Visualizes a feature map by applying a colormap and optionally saves and displays it.

    Args:
        feature_map (torch.Tensor or np.ndarray): The feature map tensor with shape [B, C, H, W] or [C, H, W].
        filename (str, optional): The filename to save the image. If None, the image will not be saved.
        colormap (int, optional): OpenCV colormap to apply. Default is cv2.COLORMAP_JET.
        display (bool, optional): If True, the image will be displayed using OpenCV's imshow. Default is False.

    Returns:
        np.ndarray: The colored feature map as a NumPy array.
    """

    # Convert tensor to numpy array if needed
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    # Remove batch dimension if present
    if feature_map.ndim == 4:
        feature_map = np.squeeze(feature_map, axis=0)  # From [B, C, H, W] to [C, H, W]

    # Handle channels
    if feature_map.shape[0] > 1:
        # Option 1: Visualize the first channel
        feature_map_to_visualize = feature_map[0, :, :]
    else:
        # If only one channel is present
        feature_map_to_visualize = feature_map[0, :, :]

    # Normalize the feature map to [0, 255]
    feature_min = feature_map_to_visualize.min()
    feature_max = feature_map_to_visualize.max()
    feature_map_to_visualize = (feature_map_to_visualize - feature_min) / (feature_max - feature_min + 1e-6)
    feature_map_to_visualize = (feature_map_to_visualize * 255).astype(np.uint8)

    # Apply colormap
    colored_feature_map = cv2.applyColorMap(feature_map_to_visualize, colormap)

    # Save the image if filename is provided
    if filename:
        cv2.imwrite(filename, colored_feature_map)

    # Display the image if requested
    if display:
        cv2.imshow('Feature Map', colored_feature_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return colored_feature_map


def visualize_feature_maps(orig_feature, world_feature, fused_world_feature, reprojected_feature, 
                           projected_feature_p, summed_feature_p, back_projected_feature_p, cam):
    # Convert feature maps to CPU and detach
    orig = orig_feature[0].max(dim=0)[0].cpu().detach().numpy()
    proj = world_feature[0].max(dim=0)[0].cpu().detach().numpy()
    reproj = reprojected_feature[0].max(dim=0)[0].cpu().detach().numpy()
    fused_proj = fused_world_feature[0].max(dim=0)[0].cpu().detach().numpy()

    proj_p = projected_feature_p[0].max(dim=0)[0].cpu().detach().numpy()
    reproj_p = back_projected_feature_p[0].max(dim=0)[0].cpu().detach().numpy()
    print(reproj_p.shape)
    fused_proj_p = summed_feature_p[0].max(dim=0)[0].cpu().detach().numpy()

    # Plot the feature maps
    fig, axes = plt.subplots(2, 4, figsize=(15, 5))
    axes[0,0].imshow(orig, cmap='viridis')
    axes[0,0].set_title('Original Feature Map')
    # axes[0,0].set_ylabel(f'Camera {cam}', fontsize=12)
    axes[0,0].axis('off')

    axes[0,1].imshow(proj, cmap='viridis')
    axes[0,1].set_title('Projected Feature Maps')
    axes[0,1].axis('off')

    axes[0,2].imshow(fused_proj, cmap='viridis')
    axes[0,2].set_title('Fused Projected Feature Maps')
    axes[0,2].axis('off')

    axes[0,3].imshow(reproj, cmap='viridis')
    axes[0,3].set_title('Reprojected Feature Map')
    axes[0,3].axis('off')

    axes[1,0].imshow(orig, cmap='viridis')
    # axes[1,0].set_title('Original Feature Map')
    axes[1,0].axis('off')

    axes[1,1].imshow(proj_p, cmap='viridis')
    # axes[1,1].set_title('Projected Feature Maps')
    axes[1,1].axis('off')

    axes[1,2].imshow(fused_proj_p, cmap='viridis')
    # axes[1,2].set_title('Fused Projected Feature Maps')
    axes[1,2].axis('off')

    axes[1,3].imshow(reproj_p, cmap='viridis')
    # axes[1,3].set_title('Reprojected Feature Map')
    axes[1,3].axis('off')

    # # Add vertical text next to the first figure
    fig.text(0.1, 0.7, f'Geometry-Based', va='center', rotation='vertical', fontsize=12)
    fig.text(0.1, 0.3, f'Geometry-free', va='center', rotation='vertical', fontsize=12)

    # Save the figure
    output_path = os.path.join('./data/results_visualizations', f"camera_{cam}_feature_maps.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def compute_iou(box1, box2):
    """Compute IoU (Intersection over Union) between two bounding boxes."""
    poly1 = Polygon([(box1[0], box1[1]), (box1[2], box1[1]),
                     (box1[2], box1[3]), (box1[0], box1[3])])
    poly2 = Polygon([(box2[0], box2[1]), (box2[2], box2[1]),
                     (box2[2], box2[3]), (box2[0], box2[3])])
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union > 0 else 0

def process_detections_across_frames(frames_detections, frames_gt, iou_threshold=0.5, subdataset='0000'):
    """
    Process frames to compute ROC and Precision-Recall curves using global assignment via the Hungarian algorithm.
    
    Parameters:
        frames_detections: dict {frame_id: [(x1, y1, x2, y2, confidence), ...]}
        frames_gt: dict {frame_id: [(x1, y1, x2, y2), ...]}
        iou_threshold: IoU threshold for a detection to be considered a true positive.
    
    Returns:
        (fpr, tpr, auc_score, precision, recall, ap)
    """
    # Lists to store detection outcomes.
    # y_true: 1 for a true positive or missed GT (false negative), 0 for a false positive.
    # y_score: the confidence score (or 0 if no detection was made for a ground truth).
    y_true = []
    y_score = []
    
    # Count total ground truths across all frames (used for recall)
    total_gt = sum(len(gt_boxes) for gt_boxes in frames_gt.values())
    
    # Process all frames (ensuring frames with only detections or only GT are handled)
    all_frames = set(frames_detections.keys()).union(frames_gt.keys())
    
    for frame_id in all_frames:
        detections = frames_detections.get(frame_id, [])
        gt_boxes = frames_gt.get(frame_id, [])
        
        # Case 1: Both detections and GT are available.
        if len(detections) > 0 and len(gt_boxes) > 0:
            num_dets = len(detections)
            num_gts = len(gt_boxes)
            cost_matrix = np.zeros((num_dets, num_gts))
            iou_matrix = np.zeros((num_dets, num_gts))
            
            # Build the cost (and IoU) matrices.
            for i, det in enumerate(detections):
                for j, gt in enumerate(gt_boxes):
                    iou = compute_iou(det[:4], gt)
                    iou_matrix[i, j] = iou
                    cost_matrix[i, j] = 1 - iou  # Lower cost for higher IoU
            
            # Solve the assignment problem.
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Track which detections and GTs have been assigned.
            assigned_detections = set()
            assigned_gts = set()
            
            # Process assigned pairs.
            for i, j in zip(row_ind, col_ind):
                assigned_detections.add(i)
                assigned_gts.add(j)
                # Use the computed IoU to decide if this detection is a TP.
                if iou_matrix[i, j] >= iou_threshold:
                    y_true.append(1)  # True positive detection.
                else:
                    y_true.append(0)  # False positive (matched but insufficient IoU).
                y_score.append(detections[i][4])
            
            # Process detections that were not assigned (all considered false positives).
            for i, det in enumerate(detections):
                if i not in assigned_detections:
                    y_true.append(0)
                    y_score.append(det[4])
                    
            # For each ground truth that was not assigned a detection, add a false negative.
            for j, gt in enumerate(gt_boxes):
                if j not in assigned_gts:
                    y_true.append(1)  # This GT was missed.
                    y_score.append(0) # No confidence associated.
        
        # Case 2: Detections exist but there are no GT boxes.
        elif detections and not gt_boxes:
            for det in detections:
                y_true.append(0)  # All detections are false positives.
                y_score.append(det[4])
        
        # Case 3: GT boxes exist but no detections were made.
        elif gt_boxes and not detections:
            for gt in gt_boxes:
                y_true.append(1)  # Missed GT (false negative).
                y_score.append(0)
    
    # Compute ROC metrics.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    
    # Compute Precision-Recall metrics.
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # Plot ROC Curve.
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, marker='.', label=f'AUC = {auc_score:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve across Frames')
    plt.legend()
    plt.savefig(f"roc_curve_{subdataset}.png")
    plt.close()
    
    # Plot Precision-Recall Curve.
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, marker='.', label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve across Frames')
    plt.legend()
    plt.savefig(f"precision_recall_curve_{subdataset}.png")
    plt.close()
    
    print("Total ground truths:", total_gt)
    print("Detection counts (TP, FP):", (sum(y_true), len(y_true) - sum(y_true)))
    print("y_score range:", min(y_score), max(y_score))
    
    return fpr, tpr, auc_score, precision, recall, ap

def compute_iou(boxA, boxB):
    """
    Compute the Intersection-over-Union (IoU) between two bounding boxes.
    Boxes are in the format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return iou

# def assign_pred_to_gt(gt_bboxes, gt_ids, pred_bboxes, pred_features, iou_threshold=0.5):
#     """
#     For a single frame, assign each ground truth object to its best predicted bounding box
#     (and corresponding feature vector) using an IoU threshold.
    
#     Parameters:
#       gt_bboxes: list of ground truth boxes (numpy arrays in [x1, y1, x2, y2])
#       gt_ids: list of ground truth IDs (integers)
#       pred_bboxes: list of predicted bounding boxes (numpy arrays in [x1, y1, x2, y2])
#       pred_features: list of predicted feature vectors corresponding to each predicted bbox.
#       iou_threshold: only assign a prediction if IoU is at least this value.
      
#     Returns:
#       A dictionary mapping gt id -> (pred_bbox, pred_feature) for matched objects.
#     """
#     assignments = {}
    
#     for gt_box, gt_id in zip(gt_bboxes, gt_ids):
#         best_iou = 0.0
#         best_pred = None
#         best_feat = None
        
#         for pred_box, pred_feat in zip(pred_bboxes, pred_features):
#             iou_val = compute_iou(gt_box, pred_box)
            
#             if iou_val > best_iou and iou_val >= iou_threshold:
#                 best_iou = iou_val
#                 best_pred = pred_box
#                 best_feat = pred_feat
                
#         if best_pred is not None:
#             assignments[gt_id] = (best_pred, best_feat)
    
#     return assignments

def assign_pred_to_gt(gt_bboxes, gt_ids, pred_bboxes, pred_features, iou_threshold=0.5):
    """
    For a single frame, assign each ground truth object to its best predicted bounding box
    (and corresponding feature vector) using an IoU threshold.
    
    Returns a dictionary mapping gt id -> (pred_index, pred_bbox, pred_feature)
    """
    assignments = {}
    for gt_box, gt_id in zip(gt_bboxes, gt_ids):
        best_iou = 0.0
        best_pred = None
        best_feat = None
        best_index = None
        for idx, (pred_box, pred_feat) in enumerate(zip(pred_bboxes, pred_features)):
            iou_val = compute_iou(gt_box, pred_box)
            if iou_val > best_iou and iou_val >= iou_threshold:
                best_iou = iou_val
                best_pred = pred_box
                best_feat = pred_feat
                best_index = idx
        if best_pred is not None:
            assignments[gt_id] = (best_index, best_pred, best_feat)
    return assignments

def compute_interframe_histogram_similarity(gt_bboxes_anchor, gt_ids_anchor, 
                                            pred_bboxes_anchor, pred_features_anchor,
                                            gt_bboxes_pair, gt_ids_pair, 
                                            pred_bboxes_pair, pred_features_pair,
                                            iou_threshold=0.5, hist_bins=16):
    """
    For two consecutive frames (anchor and pair), match predicted boxes with ground truth 
    using IoU. Then for each object (identified by a ground truth ID) that is detected in both 
    frames, compute histograms from its predicted feature vector in each frame and compute the 
    similarity (using Pearson correlation between the histograms).
    
    Parameters:
      gt_bboxes_anchor: list of gt boxes (numpy arrays) for the anchor (t-1) frame.
      gt_ids_anchor: list of gt IDs for the anchor frame.
      pred_bboxes_anchor: list of predicted boxes for the anchor frame.
      pred_features_anchor: list of predicted feature vectors for the anchor frame.
      
      gt_bboxes_pair: list of gt boxes (numpy arrays) for the pair (t) frame.
      gt_ids_pair: list of gt IDs for the pair frame.
      pred_bboxes_pair: list of predicted boxes for the pair frame.
      pred_features_pair: list of predicted feature vectors for the pair frame.
      
      iou_threshold: minimum IoU to assign a predicted box to a gt box.
      hist_bins: number of bins for the histogram.
      
    Returns:
      A list of tuples of the form:
         (gt_id, anchor_pred_box, pair_pred_box, hist_similarity)
      where hist_similarity is the Pearson correlation between the normalized histograms 
      computed from the predicted feature vectors.
    """
    # Assign predictions to gt for each frame separately.
    anchor_assignments = assign_pred_to_gt(gt_bboxes_anchor, gt_ids_anchor, 
                                           pred_bboxes_anchor, pred_features_anchor, iou_threshold)
    pair_assignments = assign_pred_to_gt(gt_bboxes_pair, gt_ids_pair, 
                                         pred_bboxes_pair, pred_features_pair, iou_threshold)
    
   
    results = []
    # For every ground truth id present in both frames, compute histogram similarity.
    for gt_id in anchor_assignments:
        if gt_id in pair_assignments:
            anchor_box, anchor_feat = anchor_assignments[gt_id]
            pair_box, pair_feat = pair_assignments[gt_id]
            
            # Flatten feature vectors if needed.
            anchor_feat = anchor_feat.flatten().cpu().numpy()
            pair_feat = pair_feat.flatten().cpu().numpy()
            print(f'anchor feature shape: {anchor_feat.shape}')
            # Compute histograms using each feature vector's min/max as the range.
            hist_anchor, _ = np.histogram(anchor_feat, bins=hist_bins,
                                          range=(np.min(anchor_feat), np.max(anchor_feat)),
                                          density=True)
            hist_pair, _ = np.histogram(pair_feat, bins=hist_bins,
                                        range=(np.min(pair_feat), np.max(pair_feat)),
                                        density=True)
            # Compute histogram similarity using Pearson correlation.
            if np.std(hist_anchor) > 1e-6 and np.std(hist_pair) > 1e-6:
                similarity = np.corrcoef(hist_anchor, hist_pair)[0, 1]
            else:
                similarity = 1.0  # Fallback if histogram variance is nearly zero
            
            results.append((gt_id, anchor_box, pair_box, similarity))
    
    return results

def compute_histogram(feature, hist_bins=16):
    """
    Compute a normalized histogram for a given feature vector.
    """
    feature = feature.flatten().cpu().numpy()
    hist, _ = np.histogram(feature, bins=hist_bins,
                           range=(np.min(feature), np.max(feature)),
                           density=True)
    return hist

def analyze_histogram_similarities(gt_bboxes_anchor, gt_ids_anchor, pred_bboxes_anchor, pred_features_anchor,
                                   gt_bboxes_pair, gt_ids_pair, pred_bboxes_pair, pred_features_pair,
                                   iou_threshold=0.5, hist_bins=16,
                                   assigned_save_path=None, cross_save_path=None):
    """
    Integrated function that:
      1. Assigns predicted detections in each frame using ground truth.
      2. Computes histogram similarity (Pearson correlation) for each assigned detection.
      3. Computes a full cross-frame histogram similarity matrix between all predicted features,
         but excludes (sets to NaN) the cells corresponding to the assigned detections.
      4. Optionally plots and saves the assigned similarity (bar chart) and cross similarity
         (heatmap) if file paths are provided.
    
    Returns:
      assigned_results: List of tuples (gt_id, anchor_pred_box, pair_pred_box, histogram_similarity)
      cross_similarity_matrix: (N_anchor x N_pair) matrix of histogram similarities with assigned pairs excluded.
    """
    # 1. Get assignments with indices.
    anchor_assignments = assign_pred_to_gt(gt_bboxes_anchor, gt_ids_anchor, pred_bboxes_anchor, pred_features_anchor, iou_threshold)
    pair_assignments   = assign_pred_to_gt(gt_bboxes_pair, gt_ids_pair, pred_bboxes_pair, pred_features_pair, iou_threshold)
    
    # For keeping track of which predicted indices (for each frame) are assigned.
    assigned_pairs = []
    assigned_results = []
    for gt_id in anchor_assignments:
        if gt_id in pair_assignments:
            anchor_index, anchor_box, anchor_feat = anchor_assignments[gt_id]
            pair_index, pair_box, pair_feat = pair_assignments[gt_id]
            # Compute histograms for both feature vectors.
            anchor_feat = anchor_feat.flatten().cpu().numpy()
            pair_feat = pair_feat.flatten().cpu().numpy()
            hist_anchor, _ = np.histogram(anchor_feat, bins=hist_bins,
                                          range=(np.min(anchor_feat), np.max(anchor_feat)),
                                          density=True)
            hist_pair, _ = np.histogram(pair_feat, bins=hist_bins,
                                        range=(np.min(pair_feat), np.max(pair_feat)),
                                        density=True)
            if np.std(hist_anchor) > 1e-6 and np.std(hist_pair) > 1e-6:
                similarity = np.corrcoef(hist_anchor, hist_pair)[0, 1]
            else:
                similarity = 1.0
            assigned_results.append((gt_id, anchor_box, pair_box, similarity))
            assigned_pairs.append((anchor_index, pair_index))
    
    # 2. Compute cross-frame histogram similarity for all predicted features.
    n_anchor = len(pred_features_anchor)
    n_pair   = len(pred_features_pair)
    cross_similarity_matrix = np.zeros((n_anchor, n_pair))
    
    hist_list_anchor = [compute_histogram(feat, hist_bins) for feat in pred_features_anchor]
    hist_list_pair   = [compute_histogram(feat, hist_bins) for feat in pred_features_pair]
    
    for i, hist_a in enumerate(hist_list_anchor):
        for j, hist_b in enumerate(hist_list_pair):
            if np.std(hist_a) > 1e-6 and np.std(hist_b) > 1e-6:
                sim = np.corrcoef(hist_a, hist_b)[0, 1]
            else:
                sim = 1.0
            cross_similarity_matrix[i, j] = sim
            
    # 3. Exclude assigned detections from cross similarity by setting them to NaN.
    for (i, j) in assigned_pairs:
        if i < n_anchor and j < n_pair:
            cross_similarity_matrix[i, j] = np.nan
            
    # 4. Plot and save results if save paths are provided.
    if assigned_save_path is not None:
        os.makedirs(os.path.dirname(assigned_save_path), exist_ok=True)
        assigned_gt_ids = [res[0] for res in assigned_results]
        assigned_similarities = [res[3] for res in assigned_results]
        plt.figure(figsize=(8,6))
        plt.bar(assigned_gt_ids, assigned_similarities, color='skyblue', alpha=0.8)
        plt.xlabel("Ground Truth ID")
        plt.ylabel("Histogram Similarity")
        plt.title("Assigned Histogram Similarity Between Matched Detections")
        plt.ylim(0, 1)
        plt.savefig(assigned_save_path, dpi=300)
        # plt.close()
    
    if cross_save_path is not None:
        os.makedirs(os.path.dirname(cross_save_path), exist_ok=True)
        plt.figure(figsize=(8,6))
        # Use a masked array so that NaN values are not plotted.
        masked_matrix = np.ma.masked_invalid(cross_similarity_matrix)
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')
        plt.imshow(masked_matrix, cmap=cmap, interpolation="nearest")
        plt.colorbar(label="Histogram Similarity (Pearson Correlation)")
        plt.xlabel("Pair Frame Box Index")
        plt.ylabel("Anchor Frame Box Index")
        plt.title("Cross-frame Histogram Similarity (Assigned Excluded)")
        plt.savefig(cross_save_path, dpi=300)
        # plt.close()
    
    return assigned_results, cross_similarity_matrix