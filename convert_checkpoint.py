"""Convert a checkpoint file to a model file.

This script extracts the model weights from a checkpoint file and saves them
in the format expected by load_multitask_model (just the state_dict).

Usage:
    python convert_checkpoint.py checkpoints/checkpoint_epoch_6.pth artifacts/multitask_unet.pth
"""

import sys
import torch
from pathlib import Path

from kymo_tracker.deeplearning.training.multitask import MultiTaskConfig

def convert_checkpoint_to_model(checkpoint_path: str, output_path: str):
    """Convert checkpoint file to model file."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load with weights_only=False to allow custom classes
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check if it's a checkpoint format (has 'model_state_dict') or direct state_dict
    if 'model_state_dict' in checkpoint:
        print(f"  Detected checkpoint format (epoch {checkpoint.get('epoch', 'unknown')})")
        model_state_dict = checkpoint['model_state_dict']
        if 'best_loss' in checkpoint:
            print(f"  Best loss: {checkpoint['best_loss']:.6f}")
    elif isinstance(checkpoint, dict) and any(k.startswith('enc') or k.startswith('denoise') or k.startswith('segment') for k in checkpoint.keys()):
        print("  Detected direct state_dict format")
        model_state_dict = checkpoint
    else:
        # Try as direct state_dict
        print("  Assuming direct state_dict format")
        model_state_dict = checkpoint
    
    # Save just the model state_dict
    print(f"Saving model to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_dict, output_path)
    print(f"✓ Successfully converted checkpoint to model file!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_checkpoint.py <checkpoint_path> <output_path>")
        print("\nExample:")
        print("  python convert_checkpoint.py checkpoints/checkpoint_epoch_6.pth artifacts/multitask_unet.pth")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_checkpoint_to_model(checkpoint_path, output_path)
