"""
CBAM (Convolutional Block Attention Module) for YOLO11
Refines a feature map with two attention stages, used after the backbone
in train/yolo11m-cbam.yaml:
- Channel attention: which feature channels matter (avg + max pooled MLP)
- Spatial attention: where in the image to look (7x7 conv over pooled maps)
"""

import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Two-stage attention mechanism:
    1. Channel Attention: Which feature channels are important (enhanced SE)
    2. Spatial Attention: Which spatial locations are important (new!)
    
    Args:
        channels (int): Number of input channels
        ratio (int): Reduction ratio for channel attention (default: 16)
        kernel_size (int): Kernel size for spatial attention conv (default: 7)
    
    Example:
        Input:  (batch=16, channels=256, height=80, width=80)
        After channel attention: Each channel weighted by importance
        After spatial attention: Each pixel location weighted by importance
        Output: (batch=16, channels=256, height=80, width=80)
    """
    
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channels = channels
        self.ratio = ratio
        
        # ============ CHANNEL ATTENTION MODULE ============
        # Similar to SE but uses both avg and max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling paths
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False)
        )
        
        # ============ SPATIAL ATTENTION MODULE ============
        # Learns where to focus in the spatial dimensions
        self.spatial_conv = nn.Conv2d(
            2,  # Input: 2 channels (max-pool + avg-pool along channel dim)
            1,  # Output: 1 channel (spatial attention map)
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass with channel + spatial attention
        
        Args:
            x: Input tensor (batch, channels, height, width)
        
        Returns:
            Attention-refined tensor (batch, channels, height, width)
        """
        # ============ STAGE 1: CHANNEL ATTENTION ============
        # Get channel statistics using both avg and max pooling
        batch, channels, height, width = x.size()
        
        # Average pool: (B, C, H, W) -> (B, C, 1, 1)
        avg_out = self.avg_pool(x).view(batch, channels)
        # Max pool: (B, C, H, W) -> (B, C, 1, 1)
        max_out = self.max_pool(x).view(batch, channels)
        
        # Pass through shared MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)
        
        # Combine and get channel attention weights
        channel_attention = self.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)
        
        # Apply channel attention
        x = x * channel_attention
        
        # ============ STAGE 2: SPATIAL ATTENTION ============
        # Get spatial statistics by pooling along channel dimension
        
        # Average across channels: (B, C, H, W) -> (B, 1, H, W)
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        # Max across channels: (B, C, H, W) -> (B, 1, H, W)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate: (B, 2, H, W)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        
        # Learn spatial attention map via convolution
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_input))
        
        # Apply spatial attention
        x = x * spatial_attention
        
        return x


# Test function
if __name__ == "__main__":
    print("Testing CBAM Implementation\n")
    
    # Same channel count as the post-backbone CBAM in yolo11m-cbam.yaml
    cbam = CBAM(channels=512, ratio=16, kernel_size=7)
    x = torch.randn(2, 512, 20, 20)
    y = cbam(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == x.shape, "CBAM must preserve the input shape"
    print()
    
    print("All tests passed!")
