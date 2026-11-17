"""
ResNet-style bottleneck block with Squeeze-and-Excitation attention
1x1 -> 3x3 -> 1x1 convolutions, SE reweighting, then a residual shortcut.
Not currently used by the YOLO model.
"""

import torch
import torch.nn as nn

from squeeze_excitation import SEBlock


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
    print("Testing SE Bottleneck Implementation\n")
    
    # Test 1: SE Bottleneck
    print("Test 1: SE Bottleneck Block")
    bottleneck = SEBottleneck(in_channels=64, out_channels=256)
    x_bn = torch.randn(2, 64, 32, 32)
    y_bn = bottleneck(x_bn)
    print(f"  Input shape:  {x_bn.shape}")
    print(f"  Output shape: {y_bn.shape}")
    print()
    
    print("All tests passed!")
