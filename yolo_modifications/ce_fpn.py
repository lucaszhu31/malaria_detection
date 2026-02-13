"""
Contrast Enhanced FPN (CE FPN) for YOLO11
Post-backbone enhancement for small object detection (parasites)

CE FPN enhances contrast between parasites and blood cell background
by learning to amplify discriminative features and suppress noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastEnhancedFPN(nn.Module):
    """
    Contrast Enhanced Feature Pyramid Network
    
    Post-backbone module that enhances contrast for small object detection.
    Works by:
    1. Adaptive channel weighting (which channels have parasite info)
    2. Contrast stretching (amplify differences)
    3. Spatial refinement (sharpen edges)
    
    Args:
        channels (int): Number of input/output channels (e.g., 512 for YOLO11m backbone output)
        reduction (int): Channel reduction ratio for bottleneck (default: 16)
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Channel attention: learn which channels have parasite features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Contrast enhancement: learnable scaling per channel
        self.contrast_scale = nn.Parameter(torch.ones(channels).view(1, channels, 1, 1))
        self.contrast_bias = nn.Parameter(torch.zeros(channels).view(1, channels, 1, 1))
        
        # Spatial refinement: sharpen edges with learnable kernel
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass: Enhance contrast for parasite detection
        
        Args:
            x: Backbone features (batch, channels, height, width)
        
        Returns:
            Enhanced features with better parasite-background contrast
        """
        # Input shape: (batch, C, H, W)
        batch, channels, h, w = x.size()
        
        # Step 1: Channel attention - learn which channels matter
        channel_weights = self.channel_attention(x)  # (batch, C, 1, 1)
        
        # Step 2: Contrast stretching
        # Amplify important channels, suppress noise
        x_enhanced = x * channel_weights
        x_enhanced = x_enhanced * self.contrast_scale + self.contrast_bias
        
        # Step 3: Spatial refinement - sharpen parasite boundaries
        x_refined = self.spatial_refine(x_enhanced)
        
        # Step 4: Residual connection - preserve original information
        x_out = x + x_refined
        
        return x_out


class CE_FPN_with_CBAM(nn.Module):
    """
    Combined Contrast Enhancement + CBAM
    
    Uses both mechanisms:
    - CE FPN: Global contrast enhancement for parasite visibility
    - CBAM: Local attention for parasite regions
    
    Args:
        channels (int): Number of input channels
    """
    
    def __init__(self, channels):
        super().__init__()
        from se_block_cbam import CBAM
        
        self.ce_fpn = ContrastEnhancedFPN(channels)
        self.cbam = CBAM(channels, ratio=16, kernel_size=7)
    
    def forward(self, x):
        """CE FPN → CBAM pipeline"""
        x = self.ce_fpn(x)    # Enhance contrast globally
        x = self.cbam(x)      # Focus attention locally
        return x
