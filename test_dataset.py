#!/usr/bin/env python3
# Test script for YOLODataset creation

import sys
from pathlib import Path

# Get the absolute path of the project directory
ROOT = Path(__file__).resolve().parent

# Add the project directory to the Python path
sys.path.insert(0, str(ROOT))

# Now try to import from the project
try:
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import LOGGER

    # Print the signature of the build_yolo_dataset function
    import inspect
    print("Function signature:", inspect.signature(build_yolo_dataset))
    
    # Create a minimal data dictionary
    data = {
        "path": str(ROOT),
        "train": "images/train",  # Just for testing, might not exist
        "val": "images/val",      # Just for testing, might not exist
        "nc": 1,
        "names": {0: "object"}
    }
    
    # Create minimal args dict
    args = {
        "imgsz": 640,
        "rect": False,
        "cache": False,
        "batch": 16,
        "single_cls": False,
        "augment": False,
        "loss": ""
    }
    
    # Try to build a dataset (this might fail if the directories don't exist, but we should get past the batch parameter issue)
    try:
        dataset = build_yolo_dataset(
            args=args,
            img_path=ROOT / "ultralytics/assets", # This directory should exist
            batch=16,
            data=data,
            mode="train",
            rect=False,
            stride=32
        )
        print("Successfully created dataset!")
    except (FileNotFoundError, AssertionError) as e:
        print(f"Expected file-related error (not a parameter issue): {e}")
    except Exception as e:
        print(f"UNEXPECTED ERROR (might be parameter related): {e}")
        import traceback
        traceback.print_exc()
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc() 