from utils import *
from torchvision.transforms import transforms
from mmdet.datasets.pipelines import Compose, LoadImageFromFile, LoadAnnotations, Resize, RandomFlip, Normalize, Pad, DefaultFormatBundle, Collect, AutoAugment
from mmdet_custom.pipelines import NegativeSampleGenerator
class Config_d:
    dataset_name = "Wildtrack" # "MultiviewX" or "Wildtrack"
    mode = "test"  #  "test" or "train"

    head = 'cascade' # 'dino' or 'rcnn' or 'cascade'

    single_head = False
    num_head = 3
    num_views = 3
    gpu_num = 2
    resume = False
    start_epoch = 0
    frozen_backbone = False
    distributed = False
    blank_views = None

    experiment_name = f'Fused_nonlinearaddition_{head}head_{num_head}_views_{dataset_name}'  # or 'Joint_LF'

    root = f"./data/{dataset_name}/"
    train_annotation_path = f'{root}{dataset_name}_coco_train_anno_chunk_id.json'
    test_annotation_path = f'{root}{dataset_name}_coco_test_anno_chunk_id.json'

    image_dir = f'{root}Image_subsets/'
    if head == 'dino':
        config_file = f'configs/{dataset_name}/dino_jointDetection.py'
    elif head == 'rcnn':
        config_file = f'configs/{dataset_name}/JointDetection.py'
    elif head == 'cascade':
        config_file = f'configs/{dataset_name}/cascade_jointDetection.py'
    # config_file = f'configs/{dataset_name}/dino_jointDetection.py'
    # config_file = f'configs/{dataset_name}/cascade_jointDetection.py'
    out_dir = f'checkpoint/fine_tune_{dataset_name}_chunk_b/'

    pretrained_backbone = f'checkpoint/fine_tune_{dataset_name}_chunk_b_3x/best_bbox_mAP_epoch_10.pth'
    pretrained_network = f'./models/{experiment_name}/wildtrack.pth'

    intrinsic_camera_matrix_filenames = [
        'intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml',
        'intr_CVLab4.xml', 'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml'
    ]
    extrinsic_camera_matrix_filenames = [
        'extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml',
        'extr_CVLab4.xml', 'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml'
    ]

    # Specify the directory path
    dir_path = f"./models/{experiment_name}"

    # Camera views available
    if dataset_name == "Wildtrack":
        camera_views = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'] # for Wildtrack
    elif dataset_name == "MultiviewX":
        camera_views = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'] # for multiviewx

    # Transform settings
    # train_resized_size = (512, 911, 3)
    train_resized_size = (640,1338, 3)
    test_resized_size = (640,1338, 3)
    # test_resized_size = (512, 911, 3)
    # test_resized_size = (704,1252, 3)

    transform = {
        'train': [
            ResizeKeepRatio((640,1333), keep_ratio=True),
            transforms.ToTensor(),  # Convert to tensor before normalizing
            transforms.Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],  # Normalizes data
                                std=[58.395 / 255, 57.12 / 255, 57.375 / 255]),
            PadToSizeDivisibleBy(32)
        ],
        'test': [
            ResizeKeepRatio((640,1333), keep_ratio=True),
            transforms.ToTensor(),  # Convert to tensor before normalizing
            transforms.Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],  # Normalizes data
                                std=[58.395 / 255, 57.12 / 255, 57.375 / 255]),
            PadToSizeDivisibleBy(32)
        ]
    }

    # Training settings
    batch_size = 1
    lr = 1e-4
    milestones = [5,10,50,75]
    gamma = 0.5
    num_epochs = 20
