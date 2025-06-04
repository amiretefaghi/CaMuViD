import json
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import numpy as np
from collections import defaultdict

class JointDetectDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, train=True, 
                view_pairs=[('C1', 'C4')], num_views=2, pad_size=75, blank_views = None, rw=False):
        self.rw = rw
        self.json_file = json_file
        self.blank_views = blank_views
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.view_pairs = view_pairs  # Now view_pairs is a list of tuples
        self.num_views = num_views
        self.images, self.annotations = self.load_images()
        self.whole_images, _ = self.load_images()
        if self.rw:
            self.annotations_by_img_id, self.bbox_by_img_id, self.labels_by_img_id, self.people_ids_by_img_id, self.coords_by_img_id = self.preprocess_annotations()
        else:
            self.annotations_by_img_id, self.bbox_by_img_id, self.labels_by_img_id, self.people_ids_by_img_id = self.preprocess_annotations()
        self.images_dict = {img_info['id']: img_info for img_info in self.images}
        self.pad_size = pad_size
        # Filter images based on view pairs
        self.filter_images_by_view_pairs()

        self.img_ids = [img_info['id'] for img_info in self.images]
        self.annotations = [anno for anno in self.annotations if anno['image_id'] in self.img_ids]
        self.positive_pairs = self.generate_positive_pairs()
        # self.imgs_pairs = self.generate_imgs_pairs()
        if not train:
            self.frame_map = self.create_frame_index_map(self.images_dict)

    def create_frame_index_map(self, camera_data):
        """
        Extracts frame numbers from the camera data dictionary and maps them
        to sequential integers starting from 1, while maintaining them as strings.
        
        Args:
        camera_data (dict): A dictionary containing camera information, including file names.
        
        Returns:
        dict: A mapping from string frame numbers to sequential integers.
        """
        # Extract frame numbers as strings
        frame_numbers = set()
        for item in camera_data.values():
            file_name = item['file_name']
            frame_number = file_name.split('/')[1].split('.')[0]
            frame_numbers.add(frame_number)

        # Sort frame numbers and create a mapping to 1, 2, 3, ...
        frame_numbers_sorted = sorted(frame_numbers, key=lambda x: int(x))  # Sorting as integers for proper numerical order
        frame_map = {frame: idx + 1 for idx, frame in enumerate(frame_numbers_sorted)}

        return frame_map
    
    def filter_images_by_view_pairs(self):
        """Filter images to include only those that belong to the specified view pairs."""
        # view_set = {view for pair in self.view_pairs for view in pair}
        self.images = [img for img in self.images if img['file_name'].split('/')[0] in ['C1']]
        # print(view_set)

    def padded_bbox_from_ann(self):

        len_data = len(data)
    def __len__(self):
        return len(self.images)
        # return len(self.imgs_pairs)

    def __getitem__(self, idx):

        images = []
        anns = []
        bboxes = []
        labels = []
        masks = []
        if self.rw:
            coords = []

        anchor_image_info = self.images[idx]
        # anchor_image_info = self.imgs_pairs[idx][0]
        anchor_image_path = os.path.join(self.root_dir, anchor_image_info['file_name'])
        anchor_camera = anchor_image_info['file_name'].split('/')[0]
        anchor_image = self.load_image(anchor_image_path)

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)

        size = (anchor_image.shape[1], anchor_image.shape[2])
        # Define the image mode and size
        mode = 'RGB'  # or 'RGBA' for transparency

        # Create a new image with a white background
        color = (0, 0, 0)  # RGB for white

        self.anchor_img_id = anchor_image_info['id']
        anchor_annotations = self.annotations_by_img_id[self.anchor_img_id]
        try:
            anchor_annotations[0]['camera'] = anchor_camera
        except:
            print(anchor_annotations)
        anchor_bbox = anchor_annotations[0]['bbox']
        
        anchor_bbox = self.bbox_by_img_id[self.anchor_img_id]
        anchor_label = self.labels_by_img_id[self.anchor_img_id]

        ### padding
        padded_anchor_bbox = anchor_bbox #+ [[0,0,0,0]]*(self.pad_size - len(anchor_bbox))
        padded_anchor_label = anchor_label #+ [[1]]*(self.pad_size - len(anchor_label))
        padded_anchor_mask = [[0,0,0,0]]*len(anchor_bbox) + [[1,1,1,1]]*(self.pad_size - len(anchor_bbox))

        images.append(anchor_image)
        anns.append(anchor_annotations)
        bboxes.append(np.array(padded_anchor_bbox))
        labels.append(np.array(padded_anchor_label))
        masks.append(np.array(padded_anchor_mask))
        if self.rw:
            coords.append(self.coords_by_img_id[self.anchor_img_id])

        # self.p_idx = np.random.choice(np.arange(len(self.positive_pairs[idx])),size=self.num_views - 1,replace=False)
        self.p_idx = range(len(self.positive_pairs[idx]))

        for p_idx in self.p_idx:
            positive_pair = self.positive_pairs[idx][p_idx]
            # positive_pair = self.imgs_pairs[idx][1]
            positive_pair_camera = positive_pair['file_name'].split('/')[0]

            if self.blank_views == None or positive_pair_camera not in self.blank_views:
                pair_image_path_p = os.path.join(self.root_dir, positive_pair['file_name'])
                positive_pair_image = self.load_image(pair_image_path_p)

            elif positive_pair_camera in self.blank_views:
                positive_pair_image = Image.new(mode, size, color)

            if self.transform is not None:
                positive_pair_image = self.transform(positive_pair_image)

            pair_img_id = positive_pair['id']
            pair_annotations = self.annotations_by_img_id[pair_img_id]
            pair_annotations[0]['camera'] = positive_pair_camera

            pair_bbox = self.bbox_by_img_id[pair_img_id]
            pair_label = self.labels_by_img_id[pair_img_id]

            ### padding
            padded_pair_bbox = pair_bbox #+ [[0,0,10,10]]*(self.pad_size - len(pair_bbox))
            padded_pair_label = pair_label #+ [[1]]*(self.pad_size - len(pair_label))
            padded_pair_mask = [[0,0,0,0]]*len(pair_bbox) + [[1,1,1,1]]*(self.pad_size - len(pair_bbox))

            images.append(positive_pair_image)
            anns.append(pair_annotations)
            bboxes.append(np.array(padded_pair_bbox))
            labels.append(np.array(padded_pair_label))
            masks.append(np.array(padded_pair_mask))
            if self.rw:
                coords.append(self.coords_by_img_id[pair_img_id])
        if self.rw:
            return idx, images, bboxes, labels, masks, coords
        else:
            return idx, images, bboxes, labels, masks




            
    def generate_positive_pairs(self):
        positive_pairs = []

        for image_info in self.images:
            anchor_view, anchor_name = image_info['file_name'].split('/')
            positive_pairs_ = []
            # for view_pair in self.view_pairs:
            #     if anchor_view in view_pair:
                    # other_view = [view for view in view_pair if view != anchor_view][0]
            for img_info in self.whole_images:
                if img_info['file_name'].split('/')[1] == anchor_name and img_info['file_name'].split('/')[0] != anchor_view:

            # positive_pair = next((img_info for img_info in self.whole_images
            #                         if img_info['file_name'].split('/')[1] == anchor_name
            #                         and img_info['file_name'].split('/')[0] != anchor_view), None)
            # positive_pairs_.append(positive_pair)
                    positive_pairs_.append(img_info)

            positive_pairs.append(positive_pairs_)
            
        return positive_pairs


    def load_images(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        return data['images'], data['annotations']

    def preprocess_annotations(self):
        bbox_by_img_id = defaultdict(list)
        labels_by_img_id = defaultdict(list)
        annotations_by_img_id = defaultdict(list)
        people_ids_by_img_id = defaultdict(list)
        coords_by_img_id = defaultdict(list)
        for anno in self.annotations:
            img_id = anno['image_id']
            person_id = anno['id']
            if self.rw:
                coord = np.array(anno['coord'])
                coords_by_img_id[img_id].append(coord)
            anno['bbox'] = np.array(anno['bbox'])
            anno['bbox'][2] += anno['bbox'][0]
            anno['bbox'][3] += anno['bbox'][1]
            bbox_by_img_id[img_id].append(list(anno['bbox']))
            labels_by_img_id[img_id].append([anno['category_id']])
            annotations_by_img_id[img_id].append(anno)
            people_ids_by_img_id[img_id].append(person_id)
        if self.rw:
            return annotations_by_img_id, bbox_by_img_id, labels_by_img_id, people_ids_by_img_id, coords_by_img_id
        else:
            return annotations_by_img_id, bbox_by_img_id, labels_by_img_id, people_ids_by_img_id

    def load_image(self, image_path):
        image = Image.open(image_path)
        # #DEBUG: Just making sure channel order is not causing trouble.
        # b, g, r = image.split()
        colors = image.split()
        image = Image.merge("RGB", (colors[2], colors[1], colors[0]))
        return image
    
class DetectionJSONDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, train = True):
        self.json_file = json_file
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.images, self.annotations = self.load_images()
        self.annotations_by_img_id = self.preprocess_annotations()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor_image_info = self.images[idx]
        anchor_image_path = os.path.join(self.root_dir, anchor_image_info['file_name'])
        anchor_image = self.load_image(anchor_image_path)
        # anchor_image = np.array(anchor_image).astype(np.float64)
        # anchor_image = torch.from_numpy(anchor_image).permute((2,0,1)) / 255.0
        
        if self.transform is not None:
            anchor_image = self.transform(anchor_image)

        anchor_img_id = anchor_image_info['id']
        anchor_annotations = self.annotations_by_img_id[anchor_img_id]

        return anchor_image, anchor_annotations

    def load_images(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        return data['images'], data['annotations']

    def preprocess_annotations(self):
        annotations_by_img_id = defaultdict(list)
        for anno in self.annotations:
            img_id = anno['image_id']
            anno['bbox'] = np.array(anno['bbox'])
            anno['bbox'][2] += anno['bbox'][0]
            anno['bbox'][3] += anno['bbox'][1]
            annotations_by_img_id[img_id].append(anno)
        return annotations_by_img_id


    def load_image(self, image_path):
        image = Image.open(image_path)
        # #DEBUG: Just making sure channel order is not causing trouble.
        # print(len(image.split()))
        colors = image.split()
        image = Image.merge("RGB", (colors[2], colors[1], colors[0]))
        # image = image.convert("RGB")
        return image
    
class JSONDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, train = True, views= ['C1','C4']):
        self.json_file = json_file
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.views = views
        self.images, self.annotations = self.load_images()
        self.whole_images, _ = self.load_images()
        self.annotations_by_img_id = self.preprocess_annotations()
        self.images_dict = {img_info['id']: img_info for img_info in self.images}
        if self.train:
            self.images = [img_info for img_info in self.images if img_info['file_name'].split('/')[0] in {self.views[0], self.views[1]}]
        else:
            self.images = [img_info for img_info in self.images if img_info['file_name'].split('/')[0] in {self.views[1]}]
        self.img_ids = [img_info['id'] for img_info in self.images]
        self.annotations = [anno for anno in self.annotations if anno['image_id'] in self.img_ids]
        self.positive_pairs = self.generate_positive_pairs()
        # self.preload_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor_image_info = self.images[idx]
        anchor_image_path = os.path.join(self.root_dir, anchor_image_info['file_name'])
        anchor_image = self.load_image(anchor_image_path)

        if self.transform is not None:
            anchor_image = self.transform(anchor_image)

        anchor_img_id = anchor_image_info['id']
        anchor_annotations = self.annotations_by_img_id[anchor_img_id]

        positive_pair = self.positive_pairs[idx]
        pair_image_path_p = os.path.join(self.root_dir, positive_pair['file_name'])
        positive_pair_image = self.load_image(pair_image_path_p)

        if self.transform is not None:
            positive_pair_image = self.transform(positive_pair_image)

        pair_img_id = positive_pair['id']
        pair_annotations = self.annotations_by_img_id[pair_img_id]

        return (anchor_image, positive_pair_image), (anchor_annotations, pair_annotations)

    def load_images(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        return data['images'], data['annotations']

    def preprocess_annotations(self):
        annotations_by_img_id = defaultdict(list)
        for anno in self.annotations:
            img_id = anno['image_id']
            anno['bbox'] = np.array(anno['bbox'])
            anno['bbox'][2] += anno['bbox'][0]
            anno['bbox'][3] += anno['bbox'][1]
            annotations_by_img_id[img_id].append(anno)
        return annotations_by_img_id

    def generate_positive_pairs(self):
        positive_pairs = []

        for image_info in self.images:
            anchor_view, anchor_name  = image_info['file_name'].split('/')
            positive_pair = next((img_info for img_info in self.whole_images
                                if img_info['file_name'].split('/')[1] == anchor_name
                                and img_info['file_name'].split('/')[0] != anchor_view
                                and img_info['file_name'].split('/')[0] in {self.views[0], self.views[1]}), None)
            positive_pairs.append(positive_pair)

        return positive_pairs


    def load_image(self, image_path):
        image = Image.open(image_path)
        # #DEBUG: Just making sure channel order is not causing trouble.
        # b, g, r = image.split()
        colors = image.split()
        image = Image.merge("RGB", (colors[2], colors[1], colors[0]))
        return image