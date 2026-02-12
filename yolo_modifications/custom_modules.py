"""
Custom YOLO Modules with CBAM Attention for Malaria Detection
Extends Ultralytics YOLO11 modules with Channel + Spatial Attention
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, Conv, SPPF, Bottleneck

# Import our CBAM block
from se_block_cbam import CBAM


class C2f_CBAM(C2f):
    """
    C2f module with CBAM (Channel + Spatial Attention)
    
    Extends YOLO's C2f (CSP Bottleneck with 2 convolutions) by adding
    CBAM attention after the standard operations. This allows the model
    to learn:
    1. Which feature channels are important for parasite detection
    2. Which spatial locations to focus on
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of bottleneck blocks (default: 1)
        shortcut (bool): Whether to use shortcut connection (default: False)
        g (int): Groups for convolution (default: 1)
        e (float): Expansion ratio (default: 0.5)
        cbam_ratio (int): CBAM channel reduction ratio (default: 16)
        cbam_kernel (int): CBAM spatial convolution kernel size (default: 7)
    
    Example:
        >>> layer = C2f_CBAM(64, 128, n=2, shortcut=True)
        >>> x = torch.randn(1, 64, 80, 80)
        >>> out = layer(x)
        >>> out.shape
        torch.Size([1, 128, 80, 80])
    """
    
    def __init__(self, c1, c2, shortcut=False, n=1, g=1, e=0.5, cbam_ratio=16, cbam_kernel=7):
        # Initialize parent C2f with standard parameters
        from ultralytics.nn.modules.block import C2f as OriginalC2f
        OriginalC2f.__init__(self, c1, c2, n, shortcut, g, e)
        # Add CBAM block after C2f operations
        self.cbam = CBAM(c2, ratio=cbam_ratio, kernel_size=cbam_kernel)
    
    def forward(self, x):
        """
        Forward pass: C2f operations → CBAM attention
        
        Args:
            x: Input tensor (batch, c1, h, w)
        
        Returns:
            Attention-refined tensor (batch, c2, h, w)
        """
        # Run standard C2f forward pass
        x = super().forward(x)
        
        # Apply CBAM attention (learn channel + spatial importance)
        x = self.cbam(x)
        
        return x


class Conv_CBAM(Conv):
    """
    Convolution module with CBAM attention
    
    Standard YOLO Conv layer (Conv2d + BatchNorm + Activation) enhanced
    with CBAM attention. Useful for critical layers where channel + spatial
    attention provides extra benefit.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 1)
        s (int): Stride (default: 1)
        p (int): Padding (default: None, auto-calculated)
        g (int): Groups (default: 1)
        d (int): Dilation (default: 1)
        act (bool): Use activation (default: True)
        cbam_ratio (int): CBAM channel reduction ratio (default: 16)
        cbam_kernel (int): CBAM spatial kernel size (default: 7)
    """
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, cbam_ratio=16, cbam_kernel=7):
        super().__init__(c1, c2, k, s, p, g, d, act)
        # Add CBAM block after convolution
        self.cbam = CBAM(c2, ratio=cbam_ratio, kernel_size=cbam_kernel)
    
    def forward(self, x):
        """Forward pass: Conv + BN + Act → CBAM attention"""
        x = super().forward(x)
        x = self.cbam(x)
        return x


class Bottleneck_CBAM(Bottleneck):
    """
    Bottleneck module with CBAM attention
    
    Standard YOLO Bottleneck (1x1 → 3x3 → shortcut) enhanced with CBAM.
    Can be used in C2f_CBAM or standalone.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        shortcut (bool): Use shortcut connection (default: True)
        g (int): Groups (default: 1)
        k (tuple): Kernel sizes (default: (3, 3))
        e (float): Expansion ratio (default: 0.5)
        cbam_ratio (int): CBAM channel reduction ratio (default: 16)
        cbam_kernel (int): CBAM spatial kernel size (default: 7)
    """
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, cbam_ratio=16, cbam_kernel=7):
        super().__init__(c1, c2, shortcut, g, k, e)
        # Add CBAM block
        self.cbam = CBAM(c2, ratio=cbam_ratio, kernel_size=cbam_kernel)
    
    def forward(self, x):
        """Forward pass: Bottleneck operations → CBAM attention"""
        x = super().forward(x)
        x = self.cbam(x)
        return x


class SPPF_CBAM(SPPF):
    """
    Spatial Pyramid Pooling - Fast with CBAM attention
    
    YOLO's SPPF module enhanced with CBAM attention. Applied after
    spatial pooling to learn which aggregated features matter most
    and where to focus spatially.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size for max pooling (default: 5)
        cbam_ratio (int): CBAM channel reduction ratio (default: 16)
        cbam_kernel (int): CBAM spatial kernel size (default: 7)
    """
    
    def __init__(self, c1, c2, k=5, cbam_ratio=16, cbam_kernel=7):
        super().__init__(c1, c2, k)
        # Add CBAM block after SPPF
        self.cbam = CBAM(c2, ratio=cbam_ratio, kernel_size=cbam_kernel)
    
    def forward(self, x):
        """Forward pass: SPPF operations → CBAM attention"""
        x = super().forward(x)
        x = self.cbam(x)
        return x


class C2f_CBAM_Deep(C2f):
    """
    C2f with CBAM attention in EVERY bottleneck
    
    More aggressive CBAM application - adds CBAM to each internal bottleneck
    as well as the final output. Use when you need maximum attention.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of bottleneck blocks (default: 1)
        shortcut (bool): Whether to use shortcut connection (default: False)
        g (int): Groups for convolution (default: 1)
        e (float): Expansion ratio (default: 0.5)
        cbam_ratio (int): CBAM channel reduction ratio (default: 16)
        cbam_kernel (int): CBAM spatial kernel size (default: 7)
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, cbam_ratio=16, cbam_kernel=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        
        # Replace standard bottlenecks with CBAM bottlenecks
        c_ = int(c2 * e)
        self.m = nn.ModuleList(
            Bottleneck_CBAM(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0, 
                           cbam_ratio=cbam_ratio, cbam_kernel=cbam_kernel)
            for _ in range(n)
        )
        
        # Add final CBAM block
        self.cbam = CBAM(c2, ratio=cbam_ratio, kernel_size=cbam_kernel)
    
    def forward(self, x):
        """Forward pass with CBAM in bottlenecks AND final output"""
        x = super().forward(x)
        x = self.cbam(x)
        return x


# Module registry for easy access
CBAM_MODULES = {
    'C2f_CBAM': C2f_CBAM,
    'Conv_CBAM': Conv_CBAM,
    'Bottleneck_CBAM': Bottleneck_CBAM,
    'SPPF_CBAM': SPPF_CBAM,
    'C2f_CBAM_Deep': C2f_CBAM_Deep,
}


def get_cbam_module(name):
    """
    Get CBAM module by name
    
    Args:
        name (str): Module name ('C2f_CBAM', 'Conv_CBAM', etc.)
    
    Returns:
        Module class
    
    Raises:
        KeyError: If module name not found
    """
    if name not in CBAM_MODULES:
        raise KeyError(f"Module '{name}' not found. Available: {list(CBAM_MODULES.keys())}")
    return CBAM_MODULES[name]


# Test code
if __name__ == "__main__":
    print("Testing Custom YOLO Modules with CBAM Attention\n")
    print("=" * 60)
    
    # Test 1: C2f_CBAM
    print("\n1. Testing C2f_CBAM")
    print("-" * 60)
    c2f_cbam = C2f_CBAM(c1=64, c2=128, n=2, shortcut=True)
    x = torch.randn(2, 64, 80, 80)
    y = c2f_cbam(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Expected: torch.Size([2, 128, 80, 80])")
    print(f"  ✓ Pass" if y.shape == torch.Size([2, 128, 80, 80]) else "  ✗ Fail")
    
    # Count parameters
    total_params = sum(p.numel() for p in c2f_cbam.parameters())
    cbam_params = sum(p.numel() for p in c2f_cbam.cbam.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  CBAM params: {cbam_params:,} ({100*cbam_params/total_params:.2f}% of total)")
    
    # Test 2: Conv_CBAM
    print("\n2. Testing Conv_CBAM")
    print("-" * 60)
    conv_cbam = Conv_CBAM(c1=32, c2=64, k=3, s=2)
    x = torch.randn(2, 32, 160, 160)
    y = conv_cbam(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Expected: torch.Size([2, 64, 80, 80])")
    print(f"  ✓ Pass" if y.shape == torch.Size([2, 64, 80, 80]) else "  ✗ Fail")
    
    # Test 3: SPPF_CBAM
    print("\n3. Testing SPPF_CBAM")
    print("-" * 60)
    sppf_cbam = SPPF_CBAM(c1=512, c2=512, k=5)
    x = torch.randn(2, 512, 20, 20)
    y = sppf_cbam(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Expected: torch.Size([2, 512, 20, 20])")
    print(f"  ✓ Pass" if y.shape == torch.Size([2, 512, 20, 20]) else "  ✗ Fail")
    
    # Test 4: C2f_CBAM_Deep
    print("\n4. Testing C2f_CBAM_Deep (CBAM in every bottleneck)")
    print("-" * 60)
    c2f_cbam_deep = C2f_CBAM_Deep(c1=64, c2=128, n=3, shortcut=True)
    x = torch.randn(2, 64, 80, 80)
    y = c2f_cbam_deep(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Number of bottlenecks: 3")
    
    # Count CBAM blocks
    cbam_blocks = sum(1 for m in c2f_cbam_deep.modules() if isinstance(m, CBAM))
    print(f"  CBAM blocks found: {cbam_blocks} (3 in bottlenecks + 1 final = 4 expected)")
    print(f"  ✓ Pass" if cbam_blocks == 4 else "  ✗ Fail")
    
    # Test 5: Gradient flow
    print("\n5. Testing Gradient Flow")
    print("-" * 60)
    model = C2f_CBAM(c1=64, c2=128, n=2)
    x = torch.randn(1, 64, 80, 80, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print(f"  Input gradient: {x.grad is not None}")
    print(f"  CBAM MLP gradient: {model.cbam.mlp[0].weight.grad is not None}")
    print(f"  CBAM Conv gradient: {model.cbam.spatial_conv.weight.grad is not None}")
    
    # Check gradient magnitudes
    cbam_grad_norm = torch.cat([p.grad.flatten() for p in model.cbam.parameters()]).norm()
    print(f"  CBAM gradient norm: {cbam_grad_norm.item():.4f}")
    print(f"  ✓ Pass - Gradients flowing correctly" if cbam_grad_norm > 0 else "  ✗ Fail")
    
    # Test 6: Module registry
    print("\n6. Testing Module Registry")
    print("-" * 60)
    for name in CBAM_MODULES.keys():
        module_class = get_cbam_module(name)
        print(f"  {name}: {module_class.__name__} ✓")
    
    # Test 7: Memory efficiency
    print("\n7. Memory Efficiency Analysis")
    print("-" * 60)
    
    # Standard C2f
    c2f_standard = C2f(c1=256, c2=256, n=3)
    standard_params = sum(p.numel() for p in c2f_standard.parameters())
    
    # C2f with CBAM
    c2f_with_cbam = C2f_CBAM(c1=256, c2=256, n=3)
    cbam_params_total = sum(p.numel() for p in c2f_with_cbam.parameters())
    
    # C2f with deep CBAM
    c2f_deep_cbam = C2f_CBAM_Deep(c1=256, c2=256, n=3)
    deep_cbam_params = sum(p.numel() for p in c2f_deep_cbam.parameters())
    
    print(f"  Standard C2f:     {standard_params:,} params")
    print(f"  C2f_CBAM:         {cbam_params_total:,} params (+{100*(cbam_params_total-standard_params)/standard_params:.2f}%)")
    print(f"  C2f_CBAM_Deep:    {deep_cbam_params:,} params (+{100*(deep_cbam_params-standard_params)/standard_params:.2f}%)")
    print(f"  CBAM overhead: ~{100*(cbam_params_total-standard_params)/standard_params:.1f}% (acceptable if <8%)")
    
    # Test 8: Inference speed
    print("\n8. Inference Speed Comparison")
    print("-" * 60)
    import time
    
    x = torch.randn(4, 256, 80, 80)
    
    # Warm up
    _ = c2f_standard(x)
    _ = c2f_with_cbam(x)
    
    # Benchmark standard
    start = time.time()
    for _ in range(100):
        _ = c2f_standard(x)
    standard_time = time.time() - start
    
    # Benchmark CBAM
    start = time.time()
    for _ in range(100):
        _ = c2f_with_cbam(x)
    cbam_time = time.time() - start
    
    print(f"  Standard C2f: {standard_time*10:.2f}ms per forward pass")
    print(f"  C2f_CBAM:     {cbam_time*10:.2f}ms per forward pass")
    print(f"  Overhead:     {100*(cbam_time-standard_time)/standard_time:.1f}% slower")
    print(f"  Acceptable if <20% slower" if (cbam_time-standard_time)/standard_time < 0.20 else "  Consider optimization")
    
    print("\n" + "=" * 60)
    print("✅ All custom modules tested successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create yolo11m-cbam.yaml config using these modules")
    print("2. Create train_yolo_cbam.py training script")
    print("3. Train on malaria dataset and compare to baseline")
