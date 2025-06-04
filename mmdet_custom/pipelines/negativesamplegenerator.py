import numpy as np
import random
from mmdet.datasets.builder import PIPELINES
from mmdet.core import bbox_overlaps
import torch

@PIPELINES.register_module()
class NegativeSampleGenerator:
    """Generate negative bounding box samples for object detection.

    This class creates new bounding boxes that do not overlap with existing ones and assigns them a different label.
    """
    def __init__(self, prob=0.5, num_neg_samples=5, neg_label=0):
        """
        Args:
            prob (float): Probability of applying the augmentation.
            num_neg_samples (int): Number of negative samples to generate.
            neg_label (int): Label to assign to negative samples.
        """
        self.prob = prob
        self.num_neg_samples = num_neg_samples
        self.neg_label = neg_label

    def __call__(self, results):
        """
        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with augmented data.
        """
        if random.random() < self.prob:
            img_shape = results['img_shape']
            gt_bboxes = results['gt_bboxes']
            gt_labels = results['gt_labels']

            neg_bboxes = []
            neg_labels = []

            for _ in range(self.num_neg_samples):
                for _ in range(100):  # Try up to 100 times to find a non-overlapping bbox
                    x1 = random.randint(0, img_shape[1] - 1)
                    y1 = random.randint(0, img_shape[0] - 1)
                    x2 = random.randint(x1 + 1, img_shape[1])
                    y2 = random.randint(y1 + 1, img_shape[0])

                    new_bbox = np.array([x1, y1, x2, y2])
                    if not self._is_overlapping(new_bbox, gt_bboxes):
                        neg_bboxes.append(new_bbox)
                        neg_labels.append(self.neg_label)
                        break

            if neg_bboxes:
                results['gt_bboxes'] = np.vstack([gt_bboxes, np.array(neg_bboxes)])
                results['gt_labels'] = np.hstack([gt_labels, np.array(neg_labels)])

        return results

    def _is_overlapping(self, bbox, gt_bboxes):
        """Check if a bounding box overlaps with any of the ground truth bounding boxes."""
        if gt_bboxes.shape[0] == 0:
            return False
        overlaps = bbox_overlaps(torch.from_numpy(np.array([bbox])), torch.from_numpy(gt_bboxes))
        return torch.any(overlaps > 0).item()