import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('Agg')  # Use the non-GUI 'Agg' backend

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
import copy
import torch
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import build_backbone, build_neck, build_head
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.detectors import BaseDetector
import torchvision.transforms.functional as TF
from utils import *
from custom_datasets_fn import * 
from mmdet.core import bbox2result
import kornia
import time

class ImageCrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(ImageCrossMultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query, mask=None):
        N, C, H, W = query.shape
        value_len, key_len, query_len = value.shape[2] * value.shape[3], key.shape[2] * key.shape[3], query.shape[2] * query.shape[3]

        # Reshape and permute the inputs to (batch_size, sequence_length, embed_size)
        value = value.permute(0, 2, 3, 1).reshape(N, value_len, C)
        key = key.permute(0, 2, 3, 1).reshape(N, key_len, C)
        query = query.permute(0, 2, 3, 1).reshape(N, query_len, C)

        # Split the embedding into self.heads different pieces
        values = self.values(value).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        # Compute the dot product between queries and keys for each head
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy levels using the softmax, so that they sum to one
        attention = torch.softmax(energy, dim=3)  # (N, heads, query_len, key_len)

        # Multiply attention by the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        # Reshape to the original spatial dimensions and pass through a final fully connected layer
        out = self.fc_out(out).view(N, H, W, C).permute(0, 3, 1, 2)  # Reshape back to (N, C, H, W)
        return out
    
class MultiLevelCrossAttention(nn.Module):
    def __init__(self, channel_sizes, heads):
        super(MultiLevelCrossAttention, self).__init__()
        self.attention_layers = nn.ModuleList([
            ImageCrossMultiHeadAttention(size, heads) for size in channel_sizes
        ])

    def forward(self, values, keys, queries, mask=None):
        outputs = []

        # Apply attention independently at each level
        for attention, value, key, query in zip(self.attention_layers, values, keys, queries):
            output = attention(value, key, query, mask)
            outputs.append(output)

        return outputs

    

def multi_level_concat(f1,f2):
        # Concatenate corresponding feature maps
    concatenated_features = [torch.cat((feature1, feature2), dim=1) for feature1, feature2 in zip(f1, f2)]

    # Convert the list of concatenated features to a tuple
    concatenated_features = tuple(concatenated_features)

    return concatenated_features

class MultiLevelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(MultiLevelConv, self).__init__()
        # Create pairs of Conv2d and ReLU activations
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU()  # ReLU activation after each convolution
            )
            for _ in range(5)  # Assuming 5 levels of feature maps
        ])

    def forward(self, feature_maps):
        # Apply each conv block (convolution followed by ReLU) to the corresponding feature map
        output_maps = [conv_block(f_map) for conv_block, f_map in zip(self.conv_blocks, feature_maps)]
        return tuple(output_maps)

class PostProcessConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(PostProcessConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        x = self.relu(x)
        return x

class SelfAttention_map(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention_map, self).__init__()
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_nonlinear = NonLinearMapping(in_channels=in_dim, out_channels=in_dim // 16)
        self.query_nonlinear = NonLinearMapping(in_channels=in_dim, out_channels=in_dim // 16)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_query = self.query_nonlinear(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_key = self.key_nonlinear(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention_map = F.softmax(energy, dim=-1)
    
        return attention_map

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 32, kernel_size=1)
        # self.key_nonlinear = NonLinearMapping(in_channels=in_dim, out_channels=in_dim // 16)
        # self.query_nonlinear = NonLinearMapping(in_channels=in_dim, out_channels=in_dim // 16)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        # proj_query = self.query_nonlinear(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        # proj_key = self.key_nonlinear(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention_map = F.softmax(energy/((width * height)**0.5), dim=-1)
        out = torch.matmul(attention_map,x.view(batch_size, -1, width * height).permute(0, 2, 1))
    
        return out.permute(0, 2, 1).view(batch_size, -1, width, height) + x


class NonLinearMapping(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super(NonLinearMapping, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # self.layernorm1 = nn.LayerNorm(out_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.elu = nn.ELU()

    def forward(self, x):
        # Expecting x to have shape (batch_size, in_channels, height, width)
        x = self.conv1(x)  # Shape: (batch_size, out_channels, height, width)
        x = self.elu(x)    # Shape: (batch_size, out_channels, height, width)
        x = self.conv2(x)  # Shape: (batch_size, out_channels, height, width)
        return x
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.inner_d_model = d_model // 8
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, self.inner_d_model)
        self.wk = nn.Linear(d_model, self.inner_d_model)
        self.wv = nn.Linear(d_model, d_model)

        # self.non_linear_mapping = NonLinearMapping(self.inner_d_model, self.inner_d_model)
        
        # Learnable parameter for summation
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.alpha = torch.tensor(0.0)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len, depth) x (..., depth, seq_len)

        dk = torch.tensor(k.shape[-1], dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        attention_weights = nn.functional.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)

        return output

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        d = q.shape[2]

        # Flatten the height and width dimensions into the sequence length
        q = q.permute(0, 2, 1)  # (batch_size, seq_len, inner_d_model)
        k = k.permute(0, 2, 1)  # (batch_size, seq_len, inner_d_model)
        v = v.permute(0, 2, 1)  # (batch_size, seq_len, d_model)

        q = self.wq(q)
        k = self.wk(k)
        # v = self.wv(v)

        # Non-linear mapping
        # q = self.non_linear_mapping(q)
        # k = self.non_linear_mapping(k)

        # Similarity calculation
        similarities = torch.matmul(q, k.transpose(-2, -1)) / (self.inner_d_model ** 0.5)  # (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Attention calculation
        output = torch.matmul(attention_weights, v)  # (batch_size, seq_len, d_model)
        
        # # alpha = torch.sigmoid(self.alpha)
        # # Weighted summation
        # print(f'alpha:{self.alpha}')
        # output = self.alpha * attended_v + v
        
        # output = output.view(batch_size, self.d_model, d)  # (batch_size, d_model, seq_len)
        output = output.permute(0, 2, 1) # (batch_size, d_model, seq_len)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_model)

    def forward(self, q, k, v):
        # Cross Attention
        attn_output = self.cross_attention(q, k, v)
        v = v + attn_output
        v = v.permute(0, 2, 1) #(batch_size, seq_len, d_model)
        v_norm = self.norm1(v)

        # MLP
        mlp_output = self.mlp(v_norm)
        v = v + mlp_output
        # v = self.norm2(v)

        return v.permute(0, 2, 1) # (batch_size, d_model, seq_len)

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        # Define convolution layers for consistency in dimensions if needed
        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, feature_maps):
        # feature_maps is a tuple of 5 tensors
        assert len(feature_maps) == 5, "Expected 5 feature maps"

        # # Determine the target size (size of the middle feature map)
        # _, _, target_h, target_w = feature_maps[2].size()
        
        resized_maps = []
        for fm in feature_maps:
            _, _, h, w = fm.size()
            
            # # Apply convolution to ensure dimensions are consistent
            # fm = self.conv(fm)
            
            # Reshape feature map to target size
            # fm = torch.reshape(fm, (fm.size(0), fm.size(1), h*w))
            fm = fm.view(fm.size(0),fm.size(1),-1)

            resized_maps.append(fm)

        # Concatenate the resized feature maps along the channel dimension
        fused_map = torch.cat(resized_maps, dim=-1)
        
        return fused_map
    
class FeatureSplit(nn.Module):
    def __init__(self):
        super(FeatureSplit, self).__init__()

    def forward(self, fused_map, original_sizes):
        # original_sizes is a list of sizes for the original feature maps
        assert len(original_sizes) == 5, "Expected 5 sizes for original feature maps"
        
        # Calculate the total number of pixels in each original feature map
        pixel_counts = [size[2] * size[3] for size in original_sizes]
        
        # Split the fused map into the original feature maps based on pixel counts
        splits = torch.split(fused_map, pixel_counts, dim=2)
        
        resized_maps = []
        for split, size in zip(splits, original_sizes):
            _, _, target_h, target_w = size
            # Reshape the split to match the original size
            # fm = torch.reshape(split,(split.size(0), split.size(1), target_h, target_w))
            fm =  split.view(split.size(0), split.size(1), target_h, target_w)
            resized_maps.append(fm)

        return tuple(resized_maps)

def RS_block(feature_maps):
    fused_map = []
    # original_sizes = []
    # features = []
    
    for i,fm in enumerate(feature_maps):
        b, c, h, w = fm.shape
        
        fm = torch.reshape(fm,(b,c,h*w))
        fused_map.append(fm)

    return tuple(fused_map)

def RSB_block(feature_maps,original_sizes):
    features = []
    for resized_map, org_size in zip(feature_maps, original_sizes):

        fm = resized_map.view(org_size[0],org_size[1],org_size[2],org_size[3])
        features.append(fm)

    return tuple(features)

# Loss function to enforce inverse relationship
def inverse_loss(A1, A2):
    identity = torch.eye(A1.size(-1)).to(A1.device)  # Assuming square attention matrices
    inverse_loss = torch.norm(torch.matmul(A1, A2) - identity)  # Batch matrix multiply A1 and A2
    return inverse_loss

class ProjectionMatrixNetwork(nn.Module):
    def __init__(self, channels, target_channels):
        super(ProjectionMatrixNetwork, self).__init__()
        self.channels = channels
        self.target_channels = target_channels
        
        # Global pooling to reduce the spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers to produce the projection matrix
        self.fc1 = nn.Linear(channels, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, target_channels * target_channels)  # Output a flattened projection matrix

        # self.prelu1 = nn.PReLU()
        # self.prelu2 = nn.PReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Global Average Pooling
        pooled_features = self.global_avg_pool(x)  # shape: (batch, channels, 1, 1)
        pooled_features = pooled_features.view(batch_size, -1)  # shape: (batch, channels)
        
        # Fully connected layers
        x = self.relu1(self.fc1(pooled_features))
        x = self.relu2(self.fc2(x))
        projection_matrix_flat = self.fc3(x)
        
        # Reshape to (batch, target_channels, channels)
        projection_matrix = projection_matrix_flat.view(batch_size, self.target_channels, self.target_channels)
        
        return projection_matrix

class ForegroundSelectorModule(nn.Module):
    def __init__(self, input_channels=512, reduced_channels=256):
        super(ForegroundSelectorModule, self).__init__()
        
        # Pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared convolution
        self.shared_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        
        # Sigmoid for attention
        self.sigmoid = nn.Sigmoid()
        
        # Final convolution to reduce channels
        self.final_conv = nn.Conv2d(input_channels, reduced_channels, kernel_size=1)
        
    def forward(self, x):
        # Get the input for skip connection
        residual = x
        
        # Apply average and max pooling
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # Shared convolution on pooled outputs
        shared_out_1 = self.shared_conv(avg_out)
        shared_out_2 = self.shared_conv(max_out)
        
        shared_out = shared_out_1 + shared_out_2
        # Sigmoid activation for attention map
        attention = self.sigmoid(shared_out)
        
        # Multiply attention with input feature map
        attention_out = attention * x
        
        # Skip connection (add original input) before final convolution
        output = self.final_conv(attention_out + residual)
        
        return output

class Refiningmodule(nn.Module):
    def __init__(self, input_channels=256):
        super(Refiningmodule, self).__init__()
        
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, input_channels, kernel_size=7, padding=3)
        self.conv_2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        
    def forward(self, x):
        # Get the input for skip connection

        output = self.relu1(self.conv_1(x))
        output = self.relu2(self.conv_2(output))

        return output
 
class CustomTwoStageDetector(BaseDetector):
    def __init__(self, backbone, neck , rpn_head=None, roi_head=None, 
                 train_cfg=None, test_cfg=None, init_cfg=None, pretrained=None, dinohead=None, num_head=2):
        super(CustomTwoStageDetector, self).__init__(init_cfg)
        self.pretrained_path = backbone.init_cfg['checkpoint']
        self.backbone = build_backbone(backbone)
        self.backbone.init_weights()
        if dinohead is not None:
            self.dinohead = True
        else:
            self.dinohead = False
        
        if neck:
            self.neck = build_neck(neck)

        self.rpn_heads = {}
        self.roi_heads = {}
        self.bbox_heads = {}
        self.num_head = num_head
        
        for i in range(self.num_head):
            if not self.dinohead:
                if rpn_head is not None:
                    rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
                    rpn_head_ = copy.deepcopy(rpn_head)
                    rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
                    setattr(self, f'rpn_head_{i+1}', build_head(rpn_head_))
                    # self.rpn_heads[f'rpn_head_{i+1}'] = build_head(rpn_head_)
                    self.with_rpn = True

                if roi_head is not None:
                    rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
                    roi_head_ = copy.deepcopy(roi_head)
                    roi_head_.update(train_cfg=rcnn_train_cfg, test_cfg=test_cfg.rcnn)
                    setattr(self, f'roi_head_{i+1}', build_head(roi_head_))

            if dinohead is not None:
                dinohead_config = copy.deepcopy(dinohead)
                self.bbox_heads[f'bbox_head_{i+1}'] = build_head(dinohead_config)
            
            setattr(self,f"refiningmodule_{i}", Refiningmodule(256))

        # setattr(self,f'nonlinear_1',NonLinearMapping(256,256))
        # setattr(self,f'nonlinear_2',NonLinearMapping(256,256))
        for i in range(2):
            # setattr(self,f'attentin_maps_{i+1}',SelfAttention_map(256))
            setattr(self,f'projection_net_{i+1}',ProjectionMatrixNetwork(256,256))
        # setattr(self,f'projection_net_1',ProjectionMatrixNetwork(256,256))
        # setattr(self,f'projection_net_2',ProjectionMatrixNetwork(256,128))
        

        self.conv = nn.Conv2d(256*self.num_head, 256, kernel_size=1)
        # setattr(self,f"fsn",ForegroundSelectorModule(input_channels=256,reduced_channels=256))
        # setattr(self,f"selfattention",SelfAttention(256))
        # for j in range(5):
        #     # setattr(self,f'cross_att_{j+1}',MultiHeadCrossAttention(256,16))
        #     setattr(self,f'cross_att_{j+1}',CrossAttentionBlock(256,16))
        #     # setattr(self,f'postconv_{i+1}_{j+1}', PostProcessingConvBlock(256))
        # for j in range(5):
        # setattr(self,f'selfatt', SelfAttention(256))
        

        self.featurefusion = FeatureFusion()
        self.featuresplit = FeatureSplit()
        
        self.cross_indxs = [[0,1],[1,0]]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._load_weights()

    def extract_subdict(self, full_dict, prefix):
        """Extract subdictionary that matches the given prefix."""
        return {k[len(prefix):]: v for k, v in full_dict.items() if k.startswith(prefix)}

    def _filter_state_dict(self, model, full_state_dict):
        # Get the current state dict of the model
        current_state_dict = model.state_dict()

        # Initialize an empty dictionary to store filtered weights
        filtered_state_dict = {}

        # Iterate over the items in the full_state_dict
        for k, v in full_state_dict.items():
            if k in current_state_dict:
                if current_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch: expected {current_state_dict[k].shape}, got {v.shape}")
            else:
                print(f"Skipping {k} as it is not found in the model's state dict")

        return filtered_state_dict


    def _load_weights(self):

        # Load the entire state dictionary from the .pth file
        full_state_dict = torch.load(self.pretrained_path)['state_dict']

        # Extract state dictionaries for different components
        neck_state_dict = self.extract_subdict(full_state_dict, 'neck.')

        if not self.dinohead:
            rpn_head_state_dict = self.extract_subdict(full_state_dict, 'rpn_head.')
            roi_head_state_dict = self.extract_subdict(full_state_dict, 'roi_head.')
        else:
            head_state_dict = self.extract_subdict(full_state_dict, 'bbox_head.')

        neck_filtered_state_dict = self._filter_state_dict(self.neck, neck_state_dict)
        # Load the state_dict into the model components
        self.neck.load_state_dict(neck_filtered_state_dict,strict=False)

        if not self.dinohead:
            for i in range(self.num_head):
                rpn_key = f'rpn_head_{i+1}'
                roi_key = f'roi_head_{i+1}'
                # if rpn_key in self.rpn_heads:
                rpn_filtered_state_dict = self._filter_state_dict(getattr(self, rpn_key), rpn_head_state_dict)
                getattr(self, rpn_key).load_state_dict(rpn_filtered_state_dict, strict=False)
                # if roi_key in self.roi_heads:
                roi_filtered_state_dict = self._filter_state_dict(getattr(self, roi_key), roi_head_state_dict)
                getattr(self, roi_key).load_state_dict(roi_filtered_state_dict, strict=False)
        else:
            for i in range(self.num_head):
                bbox_key = f'bbox_head_{i+1}'
                if bbox_key in self.bbox_heads:
                    filtered_state_dict = self._filter_state_dict(self.bbox_heads[bbox_key], head_state_dict)
                    self.bbox_heads[bbox_key].load_state_dict(filtered_state_dict, strict=False)

        print("Weights loaded successfully!")

    @auto_fp16(apply_to=('images',))

    def extract_feat(self, images):
        # Ensure there are at least two images
        assert len(images) > 1, "At least two images are required"
        # Extract features from the backbone for each image

        B = images[0].shape[0]  # Batch size
        imgs_result = []
        reprojected_features = []
        start = time.time()
        features = [self.backbone(img) for img in images]

        # If a neck is used, apply it to each set of features
        if self.neck is not None:
            features = [self.neck(f) for f in features]
            # features = self.neck(concatenated_features)

        end = time.time()
        print(f"Average timing for backbone : {((end - start)/7)*1000}")
        # world_features = []
        # for cam in range(len(images)):
        #     world_features_ = []
        #     fused_world_feature = 0
        #     for i in range(len(features[cam])):
        #         # Step 4: Use only the last level of the feature map
        #         # last_level_feature = features[cam][i].to('cuda:0')
        #         # print(f'org feature shape: {last_level_feature.shape}')
        #         # Step 5: Camera calibration and projection to world space
        #         proj_mat = proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
        #         reducedgrid_shape = list(map(lambda x: int(x / 5), [480, 1440]))
        #         world_feature = kornia.geometry.transform.warp_perspective(features[cam][i], proj_mat, reducedgrid_shape)
        #         fused_world_feature += world_feature
        #         world_features_.append(fused_world_feature)
        #     world_features.append(world_features_)

        # summed_features_c = []

        # # Iterate over the length of one of the feature sets
        # for i in range(len(world_features[0])):
        #     # Sum the corresponding elements across all split features
        #     summed_feature_c = sum(f[i] for f in world_features)
        #     # summed_feature = getattr(self,f'selfattention').forward(summed_feature)
        #     # visualize_feature_map(summed_feature,filename = f'./data/DuViDA/feature_maps/summed_feature_map_{i}.png')
        #     summed_features_c.append(summed_feature_c)

        # Ensure each feature set is a tuple of feature maps
        features = [f if isinstance(f, tuple) else (f,) for f in features]

        start_t = time.time()

        projected_features = []
        # projectedback_features = []
        attention_maps1 = []
        attention_maps2 = []
        timing_p_t = 0
        for i in range(self.num_head):
            projected_features_ = []
            # projectedback_features_ = []
            attention_maps1_ = []
            attention_maps2_ = []
            timing_p = 0
            for j in range(len(features[i])):
                batch, C, H, W = features[i][j].size()
                # visualize_feature_map(features[i][j],filename = f'./data/DuViDA/feature_maps/pre_feature_map_{i}{j}.png')
                # fsn_feature = getattr(self,f"fsn").forward(features[i][j])

                start_p = time.time()

                attention_map1 = getattr(self,f'projection_net_1').forward(features[i][j])

                flatten_feature = features[i][j].view(batch,C,H*W)
                # projected_feature = torch.matmul(flatten_feature,attention_maps1[i][j]).view(batch,C,H,W)
                projected_feature = torch.matmul(attention_map1, flatten_feature).view(batch,C,H,W)

                end_p = time.time()

                timing_p += (end_p - start_p)
                
                # projectedback_feature = torch.matmul(projected_feature.view(batch,C,H*W),attention_maps2[i][j])
                # visualize_feature_map(projected_feature,filename = f'./data/DuViDA/feature_maps/projected_feature_map_{i}{j}.png')
                projected_features_.append(projected_feature)
                # projectedback_features_.append(projectedback_feature)
                attention_map2 = getattr(self,f'projection_net_2').forward(features[i][j])

                attention_maps1_.append(attention_map1)
                attention_maps2_.append(attention_map2)
            timing_p_t += timing_p
            projected_features.append(tuple(projected_features_))
            attention_maps1.append(tuple(attention_maps1_))
            attention_maps2.append(tuple(attention_maps2_))
            # projectedback_features.append(tuple(projectedback_features_))

        print(f"average timing for projection : {(timing_p_t / 7)*1000}")

        summed_features = []
        # # Iterate over the length of one of the feature sets
        # for i in range(len(projected_features[0])):
        #     # Sum the corresponding elements across all split features
        #     summed_feature = sum(f[i] for f in projected_features)
        #     # summed_feature = getattr(self,f'selfattention').forward(summed_feature)
        #     # visualize_feature_map(summed_feature,filename = f'./data/DuViDA/feature_maps/summed_feature_map_{i}.png')
        #     summed_features.append(summed_feature)
        start_fu = time.time()
        for i in range(len(projected_features[0])):
            # Concatenate the corresponding elements across all split features along the channel dimension (dim=1)
            concatenated_feature = torch.cat([f[i] for f in projected_features], dim=1)
            
            # Apply a convolutional layer to reduce channel size
            reduced_feature = self.conv(concatenated_feature)
            
            # Optionally, you can pass it through self-attention if needed
            # reduced_feature = getattr(self, 'selfattention').forward(reduced_feature)

            # visualize_feature_map(reduced_feature, filename=f'./data/DuViDA/feature_maps/concatenated_feature_map_{i}.png')
            summed_features.append(reduced_feature)
        end_fu = time.time()

        print(f"average timing for fusion : {(end_fu - start_fu)*1000}")

        post_projected_features = []
        back_projected_features = []
        timing_r_t = 0
        for i in range(self.num_head):
            post_projected_features_ = []
            back_projected_features_ = []
            timing_r = 0
            for j in range(len(summed_features)):
                
                batch, C, H, W = summed_features[j].size()
                flatten_feature = summed_features[j].view(batch,C,H*W)
                post_projected_feature = torch.matmul(attention_maps2[i][j],flatten_feature).view(batch,C,H,W)
                # visualize_feature_map(post_projected_feature,filename = f'./data/DuViDA/feature_maps/post_feature_map_{i}{j}.png')
                back_projected_features_.append(post_projected_feature)
                start_r = time.time()
                post_projected_feature = getattr(self,f"refiningmodule_{i}").forward(post_projected_feature)
                end_r = time.time()
                timing_r += (end_r - start_r)
                post_projected_features_.append(post_projected_feature)
            timing_r_t += timing_r
            post_projected_features.append(tuple(post_projected_features_))
            back_projected_features.append(tuple(back_projected_features_))
        
        end_t = time.time()

        print(f"Average timing for refinement : {(timing_r_t/7)*1000}")

        # for cam in range(len(images)):
        #     reprojected_features_ = []
        #     for i in range(len(summed_features)):
        #         # Step 6: Project back to image space
        #         proj_mat_inv = torch.linalg.inv(proj_mats[cam].float()).repeat([B, 1, 1]).to('cuda:0')
        #         reprojected_feature = kornia.geometry.transform.warp_perspective(summed_features[i], proj_mat_inv, features[cam][i].shape[-2:])

        #         reprojected_features_.append(reprojected_feature)

        #     reprojected_features.append(tuple(reprojected_features_))
        #     # Visualize the original, projected, and reprojected feature maps
        #     visualize_feature_maps(features[cam][0].to('cuda:0'), world_features[cam][0], summed_features_c[0], reprojected_features_[0],
        #                            projected_features[cam][0],  summed_features[0], back_projected_features[cam][0], cam)

        return features, post_projected_features, projected_features, attention_maps1, attention_maps2
        # return features, reprojected_features, world_features



    def forward_train(self, imgs, img_metas, gt_bboxes, scale_factor, iteration, gt_labels = None,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None,proj_mat_1=None, proj_mat_2=None,proj_mats=None, **kwargs):
        X, concatenated_features, projected_features, attention_maps1, attention_maps2 = self.extract_feat(imgs)
        # X, concatenated_features, projected_features = self.extract_feat(imgs,proj_mats)


        losses = dict()

        proposal_lists = []
        if not self.dinohead:
            for i in range(self.num_head):
                loss_key = f'losses_{i+1}'

                # RPN forward and loss
                if self.with_rpn:
                    if loss_key not in losses.keys():
                        losses[loss_key] = dict()

                    rpn_key = f'rpn_head_{i+1}'
                    proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
                    rpn_losses, proposal_list = getattr(self, rpn_key).forward_train(
                        concatenated_features[i],
                        # X,
                        img_metas,
                        gt_bboxes[i],
                        gt_labels=None,
                        gt_bboxes_ignore=gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg,
                        **kwargs
                    )
                    losses[loss_key].update(rpn_losses)
                    proposal_lists.append(proposal_list)
                else:
                    proposal_list = proposals
                roi_key = f'roi_head_{i+1}'
                roi_losses = getattr(self, roi_key).forward_train(
                                                                concatenated_features[i], 
                                                                # X,  
                                                                img_metas, proposal_lists[i],
                                                                gt_bboxes[i], gt_labels[i], #proj_mat_1, iteration, scale_factor,
                                                                gt_bboxes_ignore, gt_masks,
                                                                **kwargs)
                losses[loss_key].update(roi_losses)

                reconstruction_loss = 0
                # for j in range(len(attention_maps1[i])):
                for j in [2,3,4]:
                    # j = -1
                        # print(f"A1:{attention_maps1[i][j].shape}")
                        # print(f"A2:{attention_maps2[i][j].shape}")
                    batch, C, H, W = X[i][j].size()
                    flatten_feature = X[i][j].view(batch,C,H*W)
                    projectedback_feature = torch.matmul(attention_maps2[i][j],projected_features[i][j].view(batch,C,H*W))
                    reconstruction_loss += 0.001 * F.mse_loss(flatten_feature, projectedback_feature)
                    # inverse_loss_value += 0.01 * inverse_loss(attention_maps1[i][j],attention_maps2[i][j])

                inverse_loss_dict = {'inverse_loss':reconstruction_loss}
                losses[loss_key].update(inverse_loss_dict)                    
        else:

            for i in range(self.num_head):

                loss_key = f'losses_{i+1}'

                if loss_key not in losses.keys():
                    losses[loss_key] = dict()

                bbox_key = f'dino_head_{i+1}'    
                bbox_losses = getattr(self,bbox_key).forward_train(concatenated_features[i], img_metas,
                                                     gt_bboxes[i], gt_labels[i],
                                                    gt_bboxes_ignore)
                losses[loss_key].update(bbox_losses)



        return losses


    def simple_test(self, imgs, img_metas, proposals=None, rescale=False, proj_mats=None):
        X, concatenated_features, projected_features, attention_maps1, attention_maps2 = self.extract_feat(imgs)
        # X, concatenated_features, projected_features = self.extract_feat(imgs,proj_mats)
        # fused_feats = tuple([getattr(self, 'selfatt').forward(feat) if j in [0,1,2] else feat for j, feat in enumerate(concatenated_features)])

        results = dict()
        proposal_lists = []
        timing_h = 0
        for i in range(self.num_head):
            result_key = f'results_{i+1}'
            rpn_key = f'rpn_head_{i+1}'
            roi_key = f'roi_head_{i+1}'
            # fused_feats = tuple([getattr(self,f'post_conv_{i+1}_{j+1}').forward(feat) for j,feat in enumerate(concatenated_features)])
            start_h = time.time()
            if not self.dinohead:
                if self.with_rpn:
                    proposal_lists.append(getattr(self, rpn_key).simple_test_rpn(concatenated_features[i], img_metas))
                else:
                    proposal_lists.append(proposals)

                result = getattr(self, roi_key).simple_test(concatenated_features[i], proposal_lists[i], img_metas, rescale=rescale)
                results[result_key] = result
            else:
                for i in range(self.num_head):
                    result_key = f'results_{i+1}'
                    result = self._simple_test(i,concatenated_features[i], img_metas, rescale=rescale)
                    results[result_key] = result
            end_h = time.time()
            timing_h += (end_h - start_h)
        
        print(f"Average timing for detection head : {(timing_h/7)*1000}")
        return results
        
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        # Implement the augmented testing logic here
        # The following is a very basic placeholder implementation

        # Process each augmented version of the image
        results = []
        for i, img in enumerate(imgs):
            img1, img2 = img  # Assuming img is a tuple of two augmented images
            x = self.extract_feat(img1, img2)
            if self.rpn_head:
                proposal_list = self.rpn_head.simple_test_rpn(x, [img_metas[i]])
            else:
                # If you don't use RPN, you need to handle proposals differently
                proposal_list = None

            if self.roi_head:
                roi_results = self.roi_head.simple_test(x, proposal_list, [img_metas[i]], rescale=rescale)
                results.append(roi_results)

        # Combine results from all augmented versions
        # The combination method depends on how you want to aggregate the results
        combined_results = self.combine_aug_results(results)

        return combined_results

    def combine_aug_results(self, results):
        """Combine results from augmented images."""
        # Placeholder implementation
        # Modify this according to how you want to combine the results
        # For example, you might average detections, perform NMS, etc.
        combined_results = results[0]  # Simplistic approach, just return results from the first augmentation
        return combined_results

    def _simple_test(self, i, feat, img_metas, rescale=False):
        results_list = getattr(self,f'dino_head_{i+1}').simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, getattr(self,f'dino_head_{i+1}').num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results


class CustomTwoStageDetector_singlehead(BaseDetector):
    def __init__(self, backbone, neck , rpn_head=None, roi_head=None, 
                 train_cfg=None, test_cfg=None, init_cfg=None, pretrained=None, dinohead=None):
        super(CustomTwoStageDetector_singlehead, self).__init__(init_cfg)
        self.pretrained_path = backbone.init_cfg['checkpoint']
        self.backbone = build_backbone(backbone)
        self.backbone.init_weights()
        if dinohead is not None:
            self.dinohead = True
        else:
            self.dinohead = False
        
        if neck:
            self.neck = build_neck(neck)
            
        # Create deep copies for each instantiation
        dinohead_config_1 = copy.deepcopy(dinohead)
        if dinohead is None:
            if rpn_head is not None:
                rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
                rpn_head_ = rpn_head.copy()
                rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
                self.rpn_head_1 = build_head(rpn_head_)
                self.with_rpn = True
            if roi_head is not None:
                # update train and test cfg here for now
                # TODO: refactor assigner & sampler
                rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
                roi_head.update(train_cfg=rcnn_train_cfg)
                roi_head.update(test_cfg=test_cfg.rcnn)
                # roi_head.pretrained = pretrained
                self.roi_head_1 = build_head(roi_head)
        if dinohead is not None: 
            # dinohead_ = dinohead.copy()
            self.bbox_head_1 = build_head(dinohead_config_1)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._load_weights()

    def extract_subdict(self, full_dict, prefix):
        """Extract subdictionary that matches the given prefix."""
        return {k[len(prefix):]: v for k, v in full_dict.items() if k.startswith(prefix)}

    def _filter_state_dict(self, model, full_state_dict):
        # Get the current state dict of the model
        current_state_dict = model.state_dict()

        # Initialize an empty dictionary to store filtered weights
        filtered_state_dict = {}

        # Iterate over the items in the full_state_dict
        for k, v in full_state_dict.items():
            if k in current_state_dict:
                if current_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch: expected {current_state_dict[k].shape}, got {v.shape}")
            else:
                print(f"Skipping {k} as it is not found in the model's state dict")

        return filtered_state_dict

    def _load_weights(self):
        # Load the entire state dictionary from the .pth file
        full_state_dict = torch.load(self.pretrained_path)['state_dict']

        # # Print keys to understand the structure
        # print(full_state_dict.keys())

        neck_state_dict = self.extract_subdict(full_state_dict, 'neck.')
        if not self.dinohead:
            rpn_head_state_dict = self.extract_subdict(full_state_dict, 'rpn_head.')
            roi_head_state_dict = self.extract_subdict(full_state_dict, 'roi_head.')
        else:
            head_state_dict = self.extract_subdict(full_state_dict, 'bbox_head.')


        # Load the state_dict into the model
        # self.backbone.load_state_dict(backbone_state_dict)
        self.neck.load_state_dict(neck_state_dict)
        if not self.dinohead:
            rpn_filtered_state_dict = self._filter_state_dict(self.rpn_head_1,rpn_head_state_dict)
            self.rpn_head_1.load_state_dict(rpn_filtered_state_dict, strict=False)
            roi_filtered_state_dict = self._filter_state_dict(self.roi_head_1,roi_head_state_dict)
            self.roi_head_1.load_state_dict(roi_filtered_state_dict, strict=False)
        else:
            filtered_state_dict = self._filter_state_dict(self.bbox_head_1,head_state_dict)
            self.bbox_head_1.load_state_dict(filtered_state_dict, strict=False)

        print("Weights loaded successfully!")

        # # Check that the weights are loaded correctly
        # for name, param in self.named_parameters():
        #     print(name, param.shape)

    @auto_fp16(apply_to=('img1', 'img2'))
    def extract_feat(self, img1, img2):
        # Extract features from the backbone
        f1 = self.backbone(img1)

        # If a neck is used, it will return a tuple of feature maps
        if self.neck is not None:
            x1 = self.neck(f1)

        else:
            x1 = f1

        # Ensure x1 and x2 are tuples of feature maps
        if not isinstance(x1, tuple):
            x1 = (x1,)

        return x1 

    def forward_train(self, img1, img2, img_metas, gt_bboxes_1, gt_bboxes_2, proj_mat_1, proj_mat_2, scale_factor, iteration, gt_labels_1 = None, gt_labels_2 = None,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        x1 = self.extract_feat(img1, img2)


        losses_1 = dict()
        # losses_2 = dict()
        if not self.dinohead:
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                self.test_cfg.rpn)
                rpn_losses_1, proposal_list_1 = self.rpn_head_1.forward_train(
                    x1,
                    img_metas,
                    gt_bboxes_1,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses_1.update(rpn_losses_1)
            else:
                proposal_list_1 = proposals

            roi_losses_1 = self.roi_head_1.forward_train(x1, img_metas, proposal_list_1,
                                                    gt_bboxes_1, gt_labels_1, #proj_mat_1, iteration, scale_factor,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
        
        else:

            bbox_losses_1 = self.bbox_head_1.forward_train(x1, img_metas,
                                                    gt_bboxes_1, gt_labels_1,
                                                    gt_bboxes_ignore)
            
        if not self.dinohead:
            losses_1.update(roi_losses_1)
        else:
            losses_1.update(bbox_losses_1)

        return losses_1

    def simple_test(self, img1, img2, img_metas, proposals=None, rescale=False):
        x1 = self.extract_feat(img1, img2)
        # x1 = self.extract_feat(img1, img2)


        # # Apply convolutions to each level of feature maps
        # conv_feature_maps_1 = self.multi_level_conv_1(x1)
        # fusion_1 = self.multi_level_conv_fusion_1(conv_feature_maps_1)

        # conv_feature_maps_2 = self.multi_level_conv_2(x2)
        # fusion_2 = self.multi_level_conv_fusion_2(conv_feature_maps_2)

        # concat_f1 = multi_level_concat(conv_feature_maps_1,fusion_2)
        # concat_f2 = multi_level_concat(conv_feature_maps_2,fusion_1)
        if not self.dinohead:
            if self.with_rpn:
                proposal_list_1 = self.rpn_head_1.simple_test_rpn(x1, img_metas)
            else:
                proposal_list_1 = proposals
            return self.roi_head_1.simple_test(x1, proposal_list_1, img_metas, rescale=rescale)
        else:
            return self._simple_test_1(x1, img_metas, rescale=rescale)
        
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        # Implement the augmented testing logic here
        # The following is a very basic placeholder implementation

        # Process each augmented version of the image
        results = []
        for i, img in enumerate(imgs):
            img1, img2 = img  # Assuming img is a tuple of two augmented images
            x = self.extract_feat(img1, img2)
            if self.rpn_head:
                proposal_list = self.rpn_head.simple_test_rpn(x, [img_metas[i]])
            else:
                # If you don't use RPN, you need to handle proposals differently
                proposal_list = None

            if self.roi_head:
                roi_results = self.roi_head.simple_test(x, proposal_list, [img_metas[i]], rescale=rescale)
                results.append(roi_results)

        # Combine results from all augmented versions
        # The combination method depends on how you want to aggregate the results
        combined_results = self.combine_aug_results(results)

        return combined_results

    def combine_aug_results(self, results):
        """Combine results from augmented images."""
        # Placeholder implementation
        # Modify this according to how you want to combine the results
        # For example, you might average detections, perform NMS, etc.
        combined_results = results[0]  # Simplistic approach, just return results from the first augmentation
        return combined_results

    def _simple_test_1(self, feat, img_metas, rescale=False):
        results_list = self.bbox_head_1.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head_1.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    def _simple_test_2(self, feat, img_metas, rescale=False):
        results_list = self.bbox_head_2.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head_1.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
