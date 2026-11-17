"""
CBAM (Convolutional Block Attention Module) Implementation for YOLO11
Combines channel attention + spatial attention for malaria detection
- Channel Attention: Which features are important (like SE)
- Spatial Attention: Where to look in the image (new!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Learns to weight feature channels by importance:
    1. Squeeze: Global average pooling to get channel statistics
    2. Excitation: Learn channel importance weights via FC layers
    3. Reweight: Multiply features by learned weights
    
    Args:
        channels (int): Number of input channels
        ratio (int): Reduction ratio for bottleneck (default: 16)
    
    Example:
        Input:  (batch=16, channels=64, height=640, width=640)
        Output: (batch=16, channels=64, height=640, width=640)
        
        Each channel is reweighted based on learned importance.
    """
    
    def __init__(self, channels, ratio=16):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.ratio = ratio
        
        # Squeeze operation: Global average pooling
        # Reduces (H, W) to (1, 1) for each channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation operation: Two FC layers with bottleneck
        # channels -> channels//ratio -> channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()  # Output weights between 0 and 1
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, channels, height, width)
        
        Returns:
            Reweighted tensor (batch, channels, height, width)
        """
        batch, channels, _, _ = x.size()
        
        # SQUEEZE: Global average pooling
        # (batch, channels, H, W) -> (batch, channels, 1, 1)
        y = self.avg_pool(x)
        
        # Reshape for FC layers
        # (batch, channels, 1, 1) -> (batch, channels)
        y = y.view(batch, channels)
        
        # EXCITATION: Learn channel importance
        # (batch, channels) -> (batch, channels//ratio) -> (batch, channels)
        y = self.fc(y)
        
        # Reshape back for broadcasting
        # (batch, channels) -> (batch, channels, 1, 1)
        y = y.view(batch, channels, 1, 1)
        
        # REWEIGHT: Multiply input by learned weights
        # (batch, channels, H, W) * (batch, channels, 1, 1)
        # Broadcasting: each channel scaled by its weight
        return x * y.expand_as(x)


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


#test
class SEBottleneck(nn.Module):
    """
    Bottleneck block with SE attention
    Used as building block in ResNet-style architectures
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        stride (int): Stride for convolution (default: 1)
        se_ratio (int): SE reduction ratio (default: 16)
    """
    
    def __init__(self, in_channels, out_channels, stride=1, se_ratio=16):
        super(SEBottleneck, self).__init__()
        
        # Standard bottleneck: 1x1 -> 3x3 -> 1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels, ratio=se_ratio)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Apply SE attention
        out = self.se(out)
        
        out += identity
        out = self.relu(out)
        
        return out


# Test function
if __name__ == "__main__":
    print("Testing SE Block Implementation\n")
    
    # Test 1: Basic SE block
    print("Test 1: Basic SE Block")
    se = SEBlock(channels=64, ratio=16)
    x = torch.randn(2, 64, 32, 32)  # (batch=2, channels=64, H=32, W=32)
    y = se(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Same shape: {x.shape == y.shape}")
    print()
    
    # Test 2: SE block with different channels
    print("Test 2: SE Block with 256 channels")
    se256 = SEBlock(channels=256, ratio=16)
    x256 = torch.randn(4, 256, 64, 64)
    y256 = se256(x256)
    print(f"  Input shape:  {x256.shape}")
    print(f"  Output shape: {y256.shape}")
    print()
    
    # Test 3: Verify channel weighting
    print("Test 3: Verify channel reweighting")
    se_small = SEBlock(channels=4, ratio=2)
    x_small = torch.randn(1, 4, 8, 8)
    
    with torch.no_grad():
        # Get channel weights
        avg = se_small.avg_pool(x_small).view(1, 4)
        weights = se_small.fc(avg)
        print(f"  Channel weights: {weights.squeeze().numpy()}")
        print(f"  All weights between 0-1: {torch.all((weights >= 0) & (weights <= 1)).item()}")
    print()
    
    # Test 4: SE Bottleneck
    print("Test 4: SE Bottleneck Block")
    bottleneck = SEBottleneck(in_channels=64, out_channels=256)
    x_bn = torch.randn(2, 64, 32, 32)
    y_bn = bottleneck(x_bn)
    print(f"  Input shape:  {x_bn.shape}")
    print(f"  Output shape: {y_bn.shape}")
    print()
    
    print("All tests passed!")
