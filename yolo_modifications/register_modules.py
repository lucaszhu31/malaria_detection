"""
YOLO Module Registry for CE FPN and CBAM

This file ensures custom modules are properly registered with YOLO parser
when training with custom architectures (yolo11m-ce-fpn-cbam.yaml)
"""

from ultralytics.nn import tasks
from ce_fpn import ContrastEnhancedFPN
from se_block_cbam import CBAM


def register_custom_modules():
    """Register custom attention modules with YOLO parser"""
    
    # Register CE FPN
    if not hasattr(tasks, 'CE_FPN'):
        setattr(tasks, 'CE_FPN', ContrastEnhancedFPN)
        print("✅ Registered CE_FPN module")
    
    # Register CBAM
    if not hasattr(tasks, 'CBAM'):
        setattr(tasks, 'CBAM', CBAM)
        print("✅ Registered CBAM module")
    
    print("✅ All custom modules registered successfully")


if __name__ == "__main__":
    register_custom_modules()
