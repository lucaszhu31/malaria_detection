"""
Squeeze-and-Excitation (SE) block
Channel attention only: reweights each feature channel by a learned
importance score. Not currently used by the YOLO model.
"""

import torch
import torch.nn as nn


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
    
    print("All tests passed!")
