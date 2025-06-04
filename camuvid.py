import os
from PIL import Image, ImageDraw
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend

from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from mmdet.models import build_detector
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
from mmcv.runner import load_checkpoint
import torchvision.transforms as transforms
import mmcv
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.ops as ops
from torch.nn.parallel import DistributedDataParallel as DDP

import random
import wandb

import cv2
import xml.etree.ElementTree as ET
from collections import defaultdict
from torch.nn.functional import normalize
import time

import re
from torchvision.datasets import VisionDataset
from sklearn.metrics.pairwise import cosine_similarity

import torch
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import build_backbone, build_neck, build_head
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.detectors import BaseDetector
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from utils import *
from custom_datasets_fn import * 
from Custom_TwoStageDetector import *
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from config import Config_d  # Import configuration
from itertools import combinations
from evaluation import *

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device(f'cuda:{local_rank}'), local_rank


def train_jointDetectmodel(model, dataloader, train_dataset, multiview_dataset, 
                            optimizer, scheduler, device,resized_size, experiment_name, local_rank,
                            num_epochs=10, head='rcnn', single_head=False, batch_size=1, gpu_num=2, distributed=True, start_epoch=0):
    """Train the model with optimizations."""
    writer = SummaryWriter(f'runs/{experiment_name}')
    model.to(device)
    model.train()

    # Constants
    original_height = 1080
    scale_factor = resized_size[0] / original_height
    camera_ids = {'C1':0, 'C2':1, 'C3':2, 'C4':3, 'C5':4, 'C6':5, 'C7':6}

    # Loop through epochs
    for epoch in range(start_epoch,num_epochs):
        if distributed:
            dataloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        iteration = 0 
        for i, (idx, images, bboxes, labels, masks) in enumerate(dataloader, 1):
            
            images = [img.to(device) for img in images]
            img_metas = [{'img_shape': resized_size, 'ori_shape': resized_size, 'pad_shape': (images[0].shape[2], images[0].shape[3]),'batch_input_shape': (images[0].shape[2],images[0].shape[3]), 'scale_factor': 1.0}] * batch_size
            # Initialize empty lists to hold processed targets and labels

            # print(labels[0].shape)
            # all_targets = []
            # all_labels = []

            bboxes, labels, masks = process_targets(bboxes, labels, masks, scale_factor, device)

            optimizer.zero_grad()
            # Initialize empty lists to hold processed targets and labels
            imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(multiview_dataset.intrinsic_matrices,
                                                                        multiview_dataset.extrinsic_matrices,
                                                                        multiview_dataset.worldgrid2worldcoord_mat)
            upsample_shape = list(map(lambda x: int(x / 5), multiview_dataset.img_shape))
            img_reduce = np.array(multiview_dataset.img_shape) / np.array(upsample_shape)
            img_zoom_mat = np.diag(np.append(img_reduce, [1]))
            # map
            map_zoom_mat = np.diag(np.append(np.ones([2]) / 5, [1]))
            # projection matrices: img feat -> map feat
            proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat).float().to(images[0].device)
                            for cam in range(multiview_dataset.num_cam)]   
            if single_head:
                loss_dict_1 = model.module.forward_train(img1 = images[0],img2 = images[1], gt_bboxes_1 = targets_1, gt_labels_1=labels_1,
                                                gt_bboxes_2 = targets_2, gt_labels_2=labels_2, img_metas = img_metas, proj_mat_1 = None, proj_mat_2 = None,scale_factor = scale_factor, iteration = epoch +1)
                loss_dict_2 = {}
            else:
                if distributed:
                    loss_dicts = model.module.forward_train(imgs = images, gt_bboxes = bboxes, gt_labels=labels, gt_bboxes_ignore=None,
                                                    img_metas = img_metas,scale_factor = scale_factor, iteration = epoch +1, proj_mats=proj_mats)
                else:
                    loss_dicts = model.forward_train(imgs = images, gt_bboxes = bboxes, gt_labels=labels, gt_bboxes_ignore=None,
                                                    img_metas = img_metas,scale_factor = scale_factor, iteration = epoch +1, proj_mats=proj_mats)                    

            # print([value for loss_dict in [loss_dict_1, loss_dict_2] for values in loss_dict.values()])
            total_loss = calculate_total_loss(loss_dicts, head=head, single_head=single_head)

            total_loss.backward()
            optimizer.step()

            # # Logging
            # if local_rank==0:
            #     log_training(writer, loss_dicts, epoch, i+1, len(dataloader),single_head=single_head)
            # if torch.rand(1) > 0.99 and local_rank==0:
            #     log_training_img(writer, model, images, img_metas, train_dataset, epoch, idx, len(dataloader), bboxes, single_head=single_head, scale_factor=scale_factor,distributed=distributed)
            running_loss += total_loss.item()
            
            # Epoch-end operations
            if (i) % 100 == 0 and local_rank==0:
                print_batch_loss(epoch, num_epochs, i+1, len(dataloader), running_loss, loss_dicts)
                # print_batch_loss(epoch, num_epochs, i, len(dataloader), running_loss, loss_dict_1)
            writer.flush()
            # iteration += batch_size * gpu_num
        # Save model every 5 epochs
        # if (epoch+1) % 5 == 0:
        if distributed:
            if local_rank == 0:
                if (epoch+1)%1 == 0:
                    torch.save(model.module.state_dict(), f'models/{experiment_name}/Joint_epoch_{epoch+1}.pth')
                
                torch.save(model.module.state_dict(), f'models/{experiment_name}/Joint_epoch_latest.pth')
            dist.barrier()
        else:
            if (epoch+1)%1 == 0:
                torch.save(model.state_dict(), f'models/{experiment_name}/Joint_epoch_{epoch+1}.pth')
            
            torch.save(model.state_dict(), f'models/{experiment_name}/Joint_epoch_latest.pth')
        # Scheduler step and logging
        scheduler.step()
        log_epoch_end(epoch, num_epochs, running_loss, len(dataloader), scheduler)

    print("Training complete")
    writer.close()

def test_jointDetectmodel(model, dataloader, dataset, dataset_sub, device, resized_size, batch_size=1, single_head=False,fov_filtering=None, pairs_combs=None, experience_name=None, blank_views=None):
    """Train the model.

    Args:
        model: The PyTorch model to be trained.
        dataloader: DataLoader for the training data.
        optimizer: Optimizer for updating the model's weights.
        loss_fn: Loss function.
        device: The device to train on.
        num_epochs: Number of epochs to train for.
    """
    model.to(device)
    model.eval()
    original_shape = (1080,1920)
    camera_ids = {'C1':0, 'C2':1, 'C3':2, 'C4':3, 'C5':4, 'C6':5, 'C7':6}

    result_dir = './results_text'
    gt_filepath = './TrackEval-master/data/gt'
    gt_filepath_mot = os.path.join(gt_filepath,'MOT17-test/')
    gt_filepath_seqmaps =  os.path.join(gt_filepath,'seqmaps')
    detected_filepath_mot = './TrackEval-master/data/trackers/MOT17-test/'
    pic_path = os.path.join('./data/DuViDA', f"results_{experience_name}/")
                            
    ensure_and_clear_directory(result_dir)
    ensure_and_clear_directory(gt_filepath_mot)
    ensure_and_clear_directory(gt_filepath_seqmaps)
    ensure_and_clear_directory(detected_filepath_mot)
    ensure_and_clear_directory(pic_path)

    all_preds_by_camera = defaultdict(dict)
    all_gts_by_camera = defaultdict(dict)
    all_idxs_by_camera = defaultdict(dict)

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_ground_truth_objects = 0
    iou_per_gt_id = {}  # To store all IoUs for each detected GT ID
    total_time = 0

    with torch.no_grad():

        for idx, (idx_, images, bboxes_, labels_, masks) in enumerate(dataloader, 1):
            # Move images to the specified device
            
            pair_len = len(dataset.positive_pairs[idx-1])
            anchor_image_info = dataset.images[idx_]
            anchor_camera = anchor_image_info['file_name'].split('/')[0]

            if anchor_camera in pairs_combs.keys():
                anchor_pairs = pairs_combs[anchor_camera]
                pair_counter = 0
                for pair in anchor_pairs:
                    imgs = []
                    anns = []
                    bboxes = []
                    people_ids = []
                    labels = []
                    p_idxs = []
                    imgs.append(images[0])
                    # anns.append(target_lists[0])
                    bboxes.append(bboxes_[0])
                    labels.append(labels_[0])
                    camera_ids_list = []
                    camera_ids_list.append(camera_ids[anchor_camera])
                    camera_views_list = []
                    camera_views_list.append(anchor_camera)
                    people_ids.append(np.array(dataset.people_ids_by_img_id[anchor_image_info['id']]))

                    for view in pair[1:]:
                        # print(f'pair view: {view}')
                        for i, p_pair in enumerate(dataset.positive_pairs[idx_]):
                            if p_pair['file_name'].split('/')[0]==view:
                                p_idx = i
                                p_idxs.append(i)
                                camera_ids_list.append(camera_ids[view])
                                camera_views_list.append(view)
                        # print(f'p_idxs: {p_idxs}')    
                        positive_pair = dataset.positive_pairs[idx_][p_idx]
                        positive_pair_camera = positive_pair['file_name'].split('/')[0]

                        if blank_views == None or positive_pair_camera not in blank_views:
                            pair_image_path_p = os.path.join(dataset.root_dir, positive_pair['file_name'])
                            positive_pair_image = dataset.load_image(pair_image_path_p)

                        elif positive_pair_camera in blank_views:
                            positive_pair_image = Image.new('RGB', (original_shape[1],original_shape[0]), (0,0,0))

                        # print(f'positive_pair: {positive_pair}')
                        # print(f'len of positive pairs: {dataset.positive_pairs[idx_]} for camera view {anchor_camera}')
                        # positive_pair = self.imgs_pairs[idx][1]

                        pair_img_id = positive_pair['id']
                        pair_annotations = dataset.annotations_by_img_id[pair_img_id]

                        pair_bbox = dataset.bbox_by_img_id[pair_img_id]
                        pair_label = dataset.labels_by_img_id[pair_img_id]                    

                        if dataset.transform is not None:
                            positive_pair_image = dataset.transform(positive_pair_image)


                        imgs.append(positive_pair_image.unsqueeze(0))
                        anns.append(pair_annotations)
                        bboxes.append(np.array(pair_bbox))
                        labels.append(np.array(pair_label))
                        people_ids.append(np.array(dataset.people_ids_by_img_id[pair_img_id]))


                    # images = [img.to(device) for img in images]
                    images = [img.to(device) for img in imgs]

                    scale_factor = resized_size[0]/original_shape[0]

                    img_metas = [dict(img_shape=resized_size,ori_shape= resized_size,pad_shape=(images[0].shape[2],images[0].shape[3]),batch_input_shape=(images[0].shape[2],images[0].shape[3]),scale_factor=1.0)] * batch_size

                    # Initialize empty lists to hold processed targets and labels
                    imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(dataset_sub.intrinsic_matrices,
                                                                                dataset_sub.extrinsic_matrices,
                                                                                dataset_sub.worldgrid2worldcoord_mat)
                    upsample_shape = list(map(lambda x: int(x / 5), dataset_sub.img_shape))
                    img_reduce = np.array(dataset_sub.img_shape) / np.array(upsample_shape)
                    img_zoom_mat = np.diag(np.append(img_reduce, [1]))
                    # map
                    map_zoom_mat = np.diag(np.append(np.ones([2]) / 5, [1]))
                    # projection matrices: img feat -> map feat
                    proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat).float().to(device)
                                    for cam in range(dataset_sub.num_cam)]   
                                                         
                    bboxes, labels, masks = process_targets(bboxes, labels, masks, scale_factor, device,evaluation=True)
                    start = time.time()
                    if not single_head:
                        results = model.simple_test(imgs=images, img_metas = img_metas, rescale=False, proj_mats=proj_mats)
                    else: 
                        results = model.simple_test(imgs=images, img_metas = img_metas, rescale=False)
                    end = time.time()
                    timing = end - start
                    total_time += timing
                    print(f'run time: {end - start}')
                        # print(f'results_1:',results_1)
                    # Load images as PIL images for drawing

                    # print(f'results: {results}')

                    anchor_image_info = dataset.images[idx_]
                    anchorview = anchor_image_info['file_name'].split('/')[0]
                    frame = anchor_image_info['file_name'].split('/')[1].split('.')[0]

                    positive_pair = dataset.positive_pairs[idx_][p_idx]
                    pairview = positive_pair['file_name'].split('/')[0]
                    
                    if single_head:
                        canvas, det_bboxes_1, det_bboxes_2 = draw_bboxes_on_image(results_1, dataset, idx_, scale_factor, targets_1, targets_2, single_head=single_head, evaluation=True, fov_filtering=fov_filtering,camera_ids=camera_ids_list)
                    else:
                        canvas, det_bboxes = draw_bboxes_on_image(results, dataset, idx_, scale_factor, bboxes, single_head=single_head, evaluation=True, fov_filtering=fov_filtering,camera_ids=camera_ids_list, p_idxs=p_idxs,people_ids=people_ids)

                    pair_name = '_'.join(pair)
                    
                    view_matches = []
                    all_gt_ids = set()
                    # Track the total number of predicted boxes across all views
                    total_pred_boxes = 0
                    all_world_coords = []
                    all_pred_world_coords = []
                    all_gt_world_coords = []

                    for j,view in enumerate(camera_views_list):
                        # if view in blank_views:
                        #     continue
                        gt_path = os.path.join(gt_filepath_mot, f'bboxes_mot{view}_pair{pair_name}')
                        pred_path = os.path.join(detected_filepath_mot, f'fused/data/bboxes_mot{view}_pair{pair_name}.txt')
                        if not os.path.exists(gt_path):
                            os.makedirs(gt_path)
                            os.makedirs(os.path.join(gt_path,'gt'))
                        if not os.path.exists(os.path.join(detected_filepath_mot, f'fused')):
                            os.makedirs(os.path.join(detected_filepath_mot, f'fused'))
                            os.makedirs(os.path.join(detected_filepath_mot, f'fused/data'))
                        if not os.path.isfile( os.path.join(gt_path,'seqinfo.ini')):
                            create_seqinfo(name=f'bboxes_mot{view}_pair{pair_name}',gt_path= gt_path)

                        # if len(det_bboxes[f'detects_{j+1}']) < 1:
                        #     print(det_bboxes[f'detects_{j+1}'])
                        #     continue

                        write_boxes_to_files(bboxes[j], det_bboxes[f'detects_{j+1}'],frame= test_dataset.frame_map[frame], people_ids=people_ids[j], gt_file= os.path.join(gt_path,'gt/gt.txt'), pred_file= pred_path)
                        eval_results = evaluate_detection_with_hungarian(pred_boxes= torch.tensor(det_bboxes[f'detects_{j+1}']).to(device),gt_boxes=bboxes[j][0], gt_ids=people_ids[j],iou_threshold=0.4)
                        world_coords = project_unmatched_boxes_to_world(unmatched_boxes=eval_results['unmatched_pred_boxes'],dataset=dataset_sub,cam_id=camera_ids[view])

                        pred_world_coords = project_unmatched_boxes_to_world(unmatched_boxes=det_bboxes[f'detects_{j+1}'],dataset=dataset_sub,cam_id=camera_ids[view])
                        # gt_world_coords = project_unmatched_boxes_to_world(unmatched_boxes=bboxes[j][0],dataset=dataset_sub,cam_id=camera_ids[view])

                        # for i, unmatched_pred in enumerate(eval_results['unmatched_pred_boxes_']):
                        #     print(f"view: {view}, frame: {frame}, index: {eval_results['unmatched_pred_indices'][i]}, FP bounding box: {unmatched_pred}, 'best iou: {eval_results['unmatched_pred_iou'][eval_results['unmatched_pred_indices'][i]]}' \n")

                        # Collect IoUs for each detected GT ID
                        for gt_id, iou in eval_results['iou_per_gt_id'].items():
                            if gt_id not in iou_per_gt_id:
                                iou_per_gt_id[gt_id] = []
                            iou_per_gt_id[gt_id].append(iou)

                        if len(world_coords)>0:
                            all_world_coords.append(np.array(world_coords).squeeze(-1))

                        # all_pred_world_coords.append(np.array(pred_world_coords).squeeze(-1))
                        # all_gt_world_coords.append(np.array(gt_world_coords).squeeze(-1))
                        # print(f"for camera view {view} : {eval_results}")
                        view_matches.append(eval_results['matched_gt_ids'])
                        all_gt_ids.update(people_ids[i])
                        # Track total number of predicted boxes
                        total_pred_boxes += len(det_bboxes[f'detects_{j+1}'])

                        if test_dataset.frame_map[frame] in all_preds_by_camera[view].keys():
                            all_preds_by_camera[view][test_dataset.frame_map[frame]] += det_bboxes[f'detects_{j+1}']
                            all_idxs_by_camera[view][test_dataset.frame_map[frame]] += [pair_counter] * len(det_bboxes[f'detects_{j+1}'])
                        else:
                            # all_preds_by_camera[view][test_dataset.frame_map[frame]]
                            all_preds_by_camera[view][test_dataset.frame_map[frame]] = det_bboxes[f'detects_{j+1}'] 
                            all_idxs_by_camera[view][test_dataset.frame_map[frame]] = [pair_counter] * len(det_bboxes[f'detects_{j+1}'])

                        if not test_dataset.frame_map[frame] in all_gts_by_camera[view].keys():
                            all_gts_by_camera[view][test_dataset.frame_map[frame]] = bboxes[j]
                    pair_counter += 1
                    
                    print(f"list of IDs: {all_gt_ids}")
                    aggregated_results = aggregate_detections_across_views(view_matches,all_gt_ids)
                    print(aggregated_results)
                    if len(all_world_coords)>0:
                        merged_points = merge_close_points(all_world_coords)
                        total_false_positives += len(merged_points)
                        print(f"FP:{len(merged_points)}")
                    else:
                        print(f"FP:0")
                        total_false_positives += 0

                    # pred_merged_points = merge_close_points(all_pred_world_coords, merge_threshold=20)
                    # # visualize_merged_points(all_pred_world_coords, pred_merged_points,filename=f"{pair_name}_merged_pred_{frame}.jpg")
                    # gt_merged_points = merge_close_points(all_gt_world_coords, merge_threshold=20)
                    # visualize_merged_points(gt_merged_points, pred_merged_points,filename=f"{pair_name}_merged_gt_pred_{frame}.jpg")
                    # print(f"number of pred merged points: {len(pred_merged_points)} and gt merged points: {len(gt_merged_points)}")
                    # print(f"pred_merged_points:{pred_merged_points}")
                    # print(f"gt_merged_points:{gt_merged_points}")
                    # Evaluate across views using the aggregated results
                    overall_evaluation = evaluate_across_views(aggregated_results, all_gt_ids, total_pred_boxes)
                    print(f"TP:{overall_evaluation['true_positives']}")
                    print(f"fn:{overall_evaluation['false_negatives']}")
                    total_true_positives += overall_evaluation['true_positives']
                    # total_false_positives += overall_evaluation['false_positives']
                    
                    total_false_negatives += overall_evaluation['false_negatives']
                    # Calculate the average IoU for each detected GT ID
  
                    # print(f"FP:{len(merged_points)}")
                    # print(f"total_FP:{total_false_positives}")
                    total_ground_truth_objects += len(all_gt_ids)
                    # # Define text properties
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_scale = 1
                    # font_color = (0, 0, 0)
                    # font_thickness = 2
                    # text_offset_x = 10
                    # text_offset_y = max_height + 30

                    # # Put names under the respective images
                    # cv2.putText(canvas, "Anchor", (text_offset_x, text_offset_y), font, font_scale, font_color, font_thickness)

                    # Save or display the concatenated image
                    concatenated_image_path = os.path.join('./data/DuViDA', f"results_{experience_name}/{pair_name}_concatenated_{frame}.jpg")
                    cv2.imwrite(concatenated_image_path, canvas)
                    print(f"Results saved to {concatenated_image_path}")
        print(f"average inference timing: {total_time/idx}")

    # Get a list of all folders in the directory
    folders = [f for f in os.listdir(gt_filepath_mot) if os.path.isdir(os.path.join(gt_filepath_mot, f))]

    # Save the folder names to a text file
    with open(os.path.join(gt_filepath_seqmaps,'MOT17-test.txt'), 'w') as file:
        file.write('name\n')  # Write 'name' as the first line
        for folder in folders:
            file.write(folder + '\n')
    # Calculate overall precision, recall, and F1-score
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou_per_gt_id = {gt_id: np.mean(iou_list) for gt_id, iou_list in iou_per_gt_id.items()}
    modp = np.mean(list(avg_iou_per_gt_id.values()))
    # Calculate MODA
    moda = 1 - (total_false_positives + total_false_negatives) / total_ground_truth_objects if total_ground_truth_objects > 0 else 0

    print({
        'MODA': moda,
        'MODP': modp,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': total_true_positives,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives
    })
    print('Folder names have been saved to ,MOT17-test.txt')

def create_combinations(n, pair_size):
    """
    Generate all combinations of a specified size from a range of numbers.

    Parameters:
    n (int): The range of numbers to create combinations from (1 to n).
    pair_size (int): The number of members in each pair.

    Returns:
    list: A list of tuples containing the combinations.
    """
    # Generate the range of numbers from 1 to n
    # numbers = range(1, n + 1)
    
    # Generate the combinations
    combs = list(combinations(n, pair_size))
    
    return combs

def create_dict_from_combinations(members_list, pair_size):
    """
    Create a dictionary where the first element of each pair is the key
    and the values are lists of combinations that start with that key.

    Parameters:
    members_list (list): The list of members to create combinations from.
    pair_size (int): The number of members in each combination.

    Returns:
    dict: A dictionary with keys as the first element of each combination
          and values as lists of combinations starting with that key.
    """
    # Generate the combinations
    combs = create_combinations(members_list, pair_size)
    
    # Create the dictionary
    comb_dict = {}
    for comb in combs:
        key = comb[0]
        if key not in comb_dict:
            comb_dict[key] = []
        comb_dict[key].append(comb)
    
    return comb_dict

if __name__ == "__main__":
    cfg = Config_d()
    if cfg.mode=='train' or cfg.mode == 'train_refiner':
        if cfg.distributed:
            device, local_rank = setup_ddp()
        else:
            device = torch.device('cuda') 
    elif cfg.mode == 'test' or cfg.mode == 'test_refiner':
        device = torch.device('cuda') 

    combinations_result = create_combinations(cfg.camera_views, cfg.num_head)
    print(f'combinations_result: {combinations_result}')

    comb_dict_result = create_dict_from_combinations(cfg.camera_views, cfg.num_head)
    print(comb_dict_result)
    
    # Check if the directory exists
    if not os.path.exists(cfg.dir_path):
        os.makedirs(cfg.dir_path)
        print(f"Directory {cfg.dir_path} was created.")
    else:
        print(f"Directory {cfg.dir_path} already exists.")

    # Initialize model from config
    model_cfg = Config.fromfile(cfg.config_file)
    if not cfg.head == 'dino':
        if cfg.single_head:
            custom_model = CustomTwoStageDetector_singlehead(
                backbone=model_cfg.model.backbone,
                neck=model_cfg.model.neck,
                rpn_head=model_cfg.model.rpn_head,
                roi_head=model_cfg.model.roi_head,
                train_cfg=model_cfg.model.get('train_cfg'),
                test_cfg=model_cfg.model.get('test_cfg'),
                pretrained=cfg.pretrained_backbone
            )
        else:
            custom_model = CustomTwoStageDetector(
                backbone=model_cfg.model.backbone,
                neck=model_cfg.model.neck,
                rpn_head=model_cfg.model.rpn_head,
                roi_head=model_cfg.model.roi_head,
                train_cfg=model_cfg.model.get('train_cfg'),
                test_cfg=model_cfg.model.get('test_cfg'),
                pretrained=cfg.pretrained_backbone,
                num_head=cfg.num_head
            )
    else:
        if cfg.single_head:
            custom_model = CustomTwoStageDetector_singlehead(
                backbone=model_cfg.model.backbone,
                neck=model_cfg.model.neck,
                dinohead=model_cfg.model.bbox_head,
                train_cfg=model_cfg.model.get('train_cfg'),
                test_cfg=model_cfg.model.get('test_cfg'),
                pretrained=cfg.pretrained_backbone
            )    
        else:

            custom_model = CustomTwoStageDetector(
                backbone=model_cfg.model.backbone,
                neck=model_cfg.model.neck,
                dinohead=model_cfg.model.bbox_head,
                train_cfg=model_cfg.model.get('train_cfg'),
                test_cfg=model_cfg.model.get('test_cfg'),
                pretrained=cfg.pretrained_backbone,
                num_head=cfg.num_head
            )
    if cfg.mode == 'train' and cfg.resume:
        pretrained_path = cfg.pretrained_network  # Adjust as necessary for test-specific weights
        state_dict = torch.load(pretrained_path)
        custom_model.load_state_dict(state_dict)
        print("Pretrained weights loaded for training.")

    if cfg.mode == 'train' or cfg.mode == 'train_refiner':
        if cfg.distributed:
            custom_model = DDP(custom_model.to(device), device_ids=[device], output_device=device)
        else:
            custom_model.to(device)  
    elif cfg.mode == 'test' or cfg.mode == 'test_refiner':
        custom_model.to(device)    

    # Load the dataset, use cfg attributes
    if cfg.dataset_name == "Wildtrack":
        dataset = Wildtrack(cfg.root)
    elif cfg.dataset_name == 'MultiviewX':
        dataset = MultiviewX(cfg.root)

    # Define your train and test transforms based on cfg.transform
    train_transform = Compose(cfg.transform['train'])
    test_transform = Compose(cfg.transform['test'])

    # Initialize optimizer and scheduler (for training only)
    if cfg.mode == 'train' or cfg.mode == 'train_refiner':
        if cfg.distributed:
            if cfg.frozen_backbone:
                # Freeze the backbone
                for param in custom_model.module.backbone.parameters():
                    param.requires_grad = False
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, custom_model.parameters()),
                lr=cfg.lr
            )
        else:
            if cfg.frozen_backbone:
                # Freeze the backbone
                for param in custom_model.backbone.parameters():
                    param.requires_grad = False
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, custom_model.parameters()),
                lr=cfg.lr
            )        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    if cfg.mode == 'test' or cfg.mode == 'train_refiner' or cfg.mode == 'test_refiner':
        pretrained_path = cfg.pretrained_network  # Adjust as necessary for test-specific weights
        state_dict = torch.load(pretrained_path, map_location=device)
        custom_model.load_state_dict(state_dict)
        print("Pretrained weights loaded for testing.")

    if cfg.mode == 'train':
        train_dataset = JointDetectDataset(cfg.train_annotation_path, cfg.image_dir, 
                                            transform=train_transform, train=True, 
                                            view_pairs=combinations_result,num_views=cfg.num_head)
        if cfg.distributed:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, sampler=DistributedSampler(train_dataset))
            # Train the model
            train_jointDetectmodel(custom_model, train_dataloader, train_dataset, dataset, optimizer, 
                                scheduler, device, cfg.train_resized_size, cfg.experiment_name, local_rank=local_rank, 
                                num_epochs=cfg.num_epochs, head=cfg.head, single_head=cfg.single_head,batch_size=cfg.batch_size,gpu_num=cfg.gpu_num, start_epoch=cfg.start_epoch)
            dist.destroy_process_group()
        else:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size)
            # Train the model
            train_jointDetectmodel(custom_model, train_dataloader, train_dataset, dataset, optimizer, 
                                scheduler, device, cfg.train_resized_size, cfg.experiment_name, local_rank=0, 
                                num_epochs=cfg.num_epochs, head=cfg.head, single_head=cfg.single_head,batch_size=cfg.batch_size,gpu_num=cfg.gpu_num, distributed=cfg.distributed, start_epoch=cfg.start_epoch)

    elif cfg.mode == 'test':
        test_dataset = JointDetectDataset(cfg.test_annotation_path, cfg.image_dir, transform=test_transform, 
                                            train=False, view_pairs=combinations_result, blank_views=cfg.blank_views)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size)
        # Test the model
        fov_filtering = FOV_Filtering(dataset)
        test_jointDetectmodel(custom_model, test_dataloader, test_dataset, dataset, device, 
                            cfg.test_resized_size, batch_size=cfg.batch_size, single_head=cfg.single_head, 
                            fov_filtering=fov_filtering, pairs_combs = comb_dict_result, experience_name=cfg.experiment_name,blank_views=cfg.blank_views)