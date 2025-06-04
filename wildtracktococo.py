import numpy as np
import os.path as osp
from os import listdir
import mmcv
from sklearn.model_selection import train_test_split
import os.path as osp
from os import listdir
import mmcv


def convert_wildtrack_to_coco_(ann, ann_dir, image_prefix):
    views = ["C1/", "C2/", "C3/", "C4/", "C5/", "C6/", "C7/"]
    views_id = {"C1/":0, "C2/":1, "C3/":2, "C4/":3, "C5/":4, "C6/":5, "C7/":6}
    
    idx = 0
    i = 0
    annotations = []
    images = []
    obj_count = 0
    for name in ann:
        data_infos = mmcv.load(osp.join(ann_dir,name))
        i+=1
        # print(i)

        for view in views:
            filename = view + name.split(sep='.')[0] + ".png"
            img_path = osp.join(image_prefix,filename)
            height, width = mmcv.imread(img_path).shape[:2]
            
            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))
            
            bboxes = []
            labels = []
            masks = []
            view_id = views_id[view]
            for obj in data_infos:
                person_id = obj["personID"]
                xmax = obj["views"][view_id]["xmax"]
                xmin = obj["views"][view_id]["xmin"]
                ymax = obj["views"][view_id]["ymax"]
                ymin = obj["views"][view_id]["ymin"]
                if xmax == -1 or xmin == -1 or ymax == -1 or ymin ==-1:
                    continue
                if xmin < -1:
                    xmin=0
                    # continue
                if xmax > 1920:
                    xmax = 1920
                    # continue
                if ymin < -1 :
                    ymin =0
                    # continue
                if ymax > 1080:
                    ymax = 1080
                    # continue
                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=1,
                    bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
                    area=(xmax - xmin) * (ymax - ymin),
                    iscrowd=1)
                annotations.append(data_anno)
                obj_count += 1                                
            
            idx += 1

    CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    )

    # Create categories list using the order of the COCO classes
    categories = [{"id": idx + 1, "name": class_name} for idx, class_name in enumerate(CLASSES)]
    
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    return coco_format_json

def convert_wildtrack_to_coco_test_train(ann_dir, out_dir, image_prefix,test_size= 0.2,datastet = "wildtrack"):
    # train_set, test_set = train_test_split(listdir(dir_),test_size= test_size,shuffle=True)
    frames_name = np.sort(listdir(ann_dir))
    train_set, test_set = frames_name[:int((1-test_size)*len(frames_name))], frames_name[int((1-test_size)*len(frames_name)):]
    train_json = convert_wildtrack_to_coco_(train_set, ann_dir, image_prefix)
    train_dir = osp.join(out_dir,f"{datastet}_coco_train_anno_chunk.json")
    mmcv.dump(train_json, train_dir)
    print("train json file saved")
    test_json = convert_wildtrack_to_coco_(test_set, ann_dir, image_prefix)
    test_dir = osp.join(out_dir,f"{datastet}_coco_test_anno_chunk.json")
    mmcv.dump(test_json, test_dir)
    print("test json file saved")

if __name__ == "__main__":

    dataset = "Wildtrack"
    ann_dir = f"./data/{dataset}/annotations_positions"
    image_prefix = f"./data/{dataset}/Image_subsets"
    out_dir = f"./data/{dataset}/"
    convert_wildtrack_to_coco_test_train(ann_dir,out_dir, image_prefix,test_size=0.0,datastet=dataset)