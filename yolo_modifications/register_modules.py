"""
YOLO Module Registry for CBAM

This file ensures custom modules are properly registered with YOLO parser
"""

from ultralytics.nn import tasks
from se_block_cbam import CBAM


def register_custom_modules():
    """Register custom attention modules with YOLO parser"""
    
    # Register CBAM
    if not hasattr(tasks, 'CBAM'):
        setattr(tasks, 'CBAM', CBAM)
        print("Registered CBAM module")
    
    print("All custom modules registered successfully")


if __name__ == "__main__":
    register_custom_modules()
