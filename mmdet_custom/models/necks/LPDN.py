import torch
import torch.nn as nn
from mmdet.models.builder import NECKS

@NECKS.register_module()
class LPDN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, num_outs=5):
        super(LPDN, self).__init__()
        # Linear projection layers for the existing feature maps
        self.layers = nn.ModuleList()
        for i in range(len(in_channels)):
            if in_channels[i] == out_channels:
                 # If in_channels equals out_channels, add an identity (pass-through)
                self.layers.append(nn.Identity())
            else:
                layer = []
                for _ in range(num_layers):
                    layer.append(nn.Conv2d(in_channels[i], out_channels, kernel_size=1))
                self.layers.append(nn.Sequential(*layer))
        
        # Additional strided convolution to downsample the smallest feature map by 2
        self.downsample_layer = nn.Conv2d(in_channels[-1], out_channels, kernel_size=1, stride=2)

    def forward(self, inputs):
        # Apply linear projection to each input feature map
        outs = []
        for i, x in enumerate(inputs):
            outs.append(self.layers[i](x))
        
        # Generate the downsampled feature map (half-size of the smallest feature map)
        downsampled_feature_map = self.downsample_layer(inputs[-1])
        outs.append(downsampled_feature_map)
        
        return tuple(outs)
