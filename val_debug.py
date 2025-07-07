import argparse
import logging
from pathlib import Path

# Configure logging to be highly verbose for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ValidationDebug")

from ultralytics import YOLO
# ❗️ Step 1: Make sure to import YOUR custom validator
from ultralytics.models.yolo.tomato.val import TomatoValidator


def main(args):
    """Main validation debugging function."""
    logger.info("--- Starting Validation Debug Script ---")

    # --- Step 2: Load your trained or partially trained model ---
    # The .pt file contains the model architecture, including your custom heads.
    try:
        model = YOLO(args.model)
        logger.info(f"Successfully loaded model from {args.model}")
    except Exception as e:
        logger.error(f"Failed to load model from {args.model}. Error: {e}")
        return

    # --- Step 3: Define arguments for the validator ---
    # These arguments will override the defaults and configure the validation run.
    validator_args = dict(
        task='tomato',  # The base task is detection
        model=args.model,
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        # Key settings for debugging:
        workers=0,      # Use 0 workers to avoid multi-processing issues
        plots=True,     # Generate plots (like confusion matrix, P/R curves)
        verbose=True,   # Print detailed information during validation
        device=args.device
    )
    logger.info(f"Validator will be run with the following arguments: {validator_args}")


    # --- Step 4: Manually create and run the validator ---
    # This mimics the logic inside the model.val() method.
    try:
        # We explicitly create an instance of YOUR TomatoValidator
        validator = TomatoValidator(args=validator_args)
        
        # The validator needs access to the model's structure and weights
        validator.model = model.model

        logger.info("Starting validator run...")
        # The validator.__call__() method kicks off the entire validation process
        metrics = validator()
        
        logger.info("--- Validation Run Completed ---")
        logger.info(f"Final Metrics: {metrics}")

    except Exception as e:
        import traceback
        logger.error("--- An error occurred during the validation process ---")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug script for YOLO Tomato Validation.")
    # --- Step 5: Configure your paths and settings here ---
    parser.add_argument("--model", type=str, default=r"D:/EdgeDownloads/best (7).pt", help="Path to your .pt model file.")
    parser.add_argument("--data", type=str, default=r"D:/TomatoDataset/roboflow-v2/data.yaml", help="Path to your data.yaml file.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for validation (use a small number).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for validation.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g., 0 or cpu.")
    
    opt = parser.parse_args()
    main(opt)