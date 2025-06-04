import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset
from mmdet.datasets import build_dataset
import mmcv
import pickle
import numpy as np

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']
# root = "/home/amir/InternImage/detection/data/Wildtrack/"
root = "/home/amir/InternImage/detection/data//"

def calculate_iou(boxA, boxB):
    # Calculate the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the area of both the prediction and ground truth rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou
def calculate_ap(precision, recall):
    # Add endpoints (0.0, 1.0) for better interpolation
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # Compute the area under the precision-recall curve (AP)
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return ap

def get_intrinsic_extrinsic_matrix(root, camera_i):
    intrinsic_camera_path = os.path.join(root, 'calibrations', 'intrinsic_zero')
    intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                            intrinsic_camera_matrix_filenames[camera_i]),
                                            flags=cv2.FILE_STORAGE_READ)
    intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
    intrinsic_params_file.release()

    extrinsic_params_file_root = ET.parse(os.path.join(root, 'calibrations', 'extrinsic',
                                                        extrinsic_camera_matrix_filenames[camera_i])).getroot()

    rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
    rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

    tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
    tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

    return intrinsic_matrix, extrinsic_matrix

def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
    image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
    world_coord = project_mat @ image_coord
    world_coord = world_coord[:2, :] / world_coord[2, :]
    return world_coord

def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.delete(project_mat, 2, 1)
    world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    image_coord = project_mat @ world_coord
    image_coord = image_coord[:2, :] / image_coord[2, :]
    return image_coord

def get_worldcoord_from_worldgrid(worldgrid):
    # datasets default unit: centimeter & origin: (-300,-900)
    grid_x, grid_y = worldgrid
    coord_x = -300 + 2.5 * grid_x
    coord_y = -900 + 2.5 * grid_y
    return np.array([coord_x, coord_y])

def get_worldgrid_from_pos( pos):
    grid_x = pos % 480
    grid_y = pos // 480
    return np.array([grid_x, grid_y], dtype=int)
    
def get_worldcoord_from_pos( pos):
    grid = get_worldgrid_from_pos(pos)
    return get_worldcoord_from_worldgrid(grid)

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


def is_coordinate_on_plane(coordinate, plane, fixed_z):
    # Unpack the plane coefficients
    A, B, C, D = plane

    # Extract the x and y values from the coordinate
    x, y = coordinate

    # Calculate the z-value using the plane equation
    z = (-A * x - B * y - D) / C

    # Set a tolerance value for floating-point comparisons
    tolerance = 1e-6

    # Check if the calculated z-value is approximately equal to the fixed z-coordinate
    return abs(z - fixed_z) < tolerance

def is_point_within_grid(point, grid_points):
    # Find the minimum and maximum x and y values of the grid points
    x_values = [p[0] for p in grid_points]
    y_values = [p[1] for p in grid_points]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    # Check if the point is within the grid boundaries
    x, y = point
    return min_x <= x <= max_x and min_y <= y <= max_y

def Obj_filter(results):
    # Set paths and filenames
    # Set paths and filenames
    config_file = 'configs/wildtrack/cascade_internimage_xl_fpn_3x_coco.py'
    results_file = '../data/Out_wildtrack.pkl'

    # Load the configuration and checkpoint
    cfg = mmcv.Config.fromfile(config_file)
    cfg.model.pretrained = None  # Avoid using pre-trained weights

    # Build the dataset
    dataset = build_dataset(cfg.data.test)
    camera_ids = {'C1':0, 'C2':1, 'C3':2, 'C4':3, 'C5':4, 'C6':5, 'C7':6}
    # Load the results from the pickle file
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    xi = np.arange(0, 480, 40)
    yi = np.arange(0, 1440, 40)
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_coord = get_worldcoord_from_worldgrid(world_grid)

    # Iterate over the dataset
    for i in range(len(dataset.data_infos)):

        # Load the image and corresponding annotation
        img_info = dataset.data_infos[i]
        img = mmcv.imread(dataset.img_prefix + img_info['filename'])
        camera_id = camera_ids[img_info['file_name'].split('/')[0]]
        ann = dataset.get_ann_info(i)
        intrinsic_mat, extrinsic_mat = get_intrinsic_extrinsic_matrix(root, camera_id)

        img_coord = get_imagecoord_from_worldcoord(world_coord, intrinsic_mat,
                                                                extrinsic_mat)
        img_coord = img_coord[:, np.where((img_coord[0] > 0) & (img_coord[1] > 0) &
                                            (img_coord[0] < 1920) & (img_coord[1] < 1080))[0]]
        img_coord = img_coord.astype(int).transpose()

        # Get the corresponding result for the current image
        result = results[i][0][0]
        # Get ground truth bounding boxes
        gt_bboxes = ann['bboxes']
        # filtered_bboxes = []
        s = 0

        for j in range(len(result)):
            point = np.array([[(result[j][0]+result[j][2])/2],[result[j][3]]],dtype=np.int32)
            world_point = get_worldcoord_from_imagecoord(point, intrinsic_mat,
                                                                extrinsic_mat)
            world_point = world_point.astype(int).transpose()[0]
            

            within_grid = is_point_within_grid(world_point, world_coord.transpose())

            if within_grid:
                if s==0:
                    filtered_bboxes = np.expand_dims(result[j],axis=0)
                    s = 1
                else:
                    filtered_bboxes = np.append(filtered_bboxes,np.expand_dims(result[j],axis=0),axis=0)
        
        results[i][0][0] = filtered_bboxes
    return results