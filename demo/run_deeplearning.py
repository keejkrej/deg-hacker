"""Stage 4: Run deep learning inference pipeline on test cases."""

import numpy as np
from pathlib import Path
import sys
import json

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from kymo_tracker.deeplearning.predict import (
    denoise_and_segment_chunked,
    create_mask_from_centers_widths,
    extract_trajectories_from_mask,
)
from kymo_tracker.deeplearning.training.multitask import load_multitask_model
from kymo_tracker.utils.device import get_default_device


def run_deeplearning_pipeline(kymograph_noisy, model, device):
    """Run deep learning denoising + locator pipeline."""
    denoised, track_params = denoise_and_segment_chunked(
        model,
        kymograph_noisy,
        device=device,
        chunk_size=16,
        overlap=8,
    )
    
    centers = track_params['centers']  # (T, N_tracks)
    widths = track_params['widths']    # (T, N_tracks)
    
    # Create segmentation mask from centers/widths
    mask, labeled_mask = create_mask_from_centers_widths(
        centers, widths, kymograph_noisy.shape
    )
    
    # Extract trajectories using the integrated function
    n_tracks = centers.shape[1] if centers.ndim > 1 else 1
    
    # First try extracting from mask
    trajectories = extract_trajectories_from_mask(
        kymograph_noisy, labeled_mask, n_tracks
    )
    
    # If all trajectories are NaN (mask extraction failed), fall back to using centers directly
    all_nan = all(np.all(np.isnan(traj)) for traj in trajectories)
    if all_nan and centers.ndim > 1:
        # Use centers as trajectories directly
        trajectories = []
        for track_idx in range(n_tracks):
            track_centers = centers[:, track_idx].copy()
            # Only include track if it has at least some valid (non-NaN) values
            if np.any(~np.isnan(track_centers)):
                trajectories.append(track_centers)
        
        # If still no valid trajectories, add at least one empty one
        if not trajectories:
            trajectories.append(np.full(kymograph_noisy.shape[0], np.nan))
    
    return {
        'denoised': denoised,
        'mask': mask,
        'labeled_mask': labeled_mask,
        'trajectories': trajectories,
        'centers': centers,
        'widths': widths,
    }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run deep learning inference pipeline")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/demo_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="demo/data",
        help="Directory containing test cases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo/results/deeplearning",
        help="Directory to save deep learning results",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STAGE 4: Running Deep Learning Pipeline")
    print("=" * 70)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = get_default_device()
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    model = load_multitask_model(str(model_path), device=device, max_tracks=3)
    print("Model loaded successfully")
    
    # Load metadata
    metadata_path = data_dir / "cases_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        cases_metadata = json.load(f)
    
    print(f"Found {len(cases_metadata)} test cases")
    
    # Process each case
    results = []
    for i, case_meta in enumerate(cases_metadata, 1):
        print(f"\nProcessing case {i}/{len(cases_metadata)}: {case_meta['name']}...")
        
        # Load noisy kymograph
        noisy_path = Path(case_meta['noisy_path'])
        if not noisy_path.exists():
            raise FileNotFoundError(f"Test case file not found: {noisy_path}")
        
        noisy_kymo = np.load(noisy_path)
        
        # Run deep learning pipeline
        print("  Running deep learning pipeline...")
        result = run_deeplearning_pipeline(noisy_kymo, model, device)
        
        # Save results
        case_output_dir = output_dir / f"case_{i:02d}"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(case_output_dir / "denoised.npy", result['denoised'])
        np.save(case_output_dir / "mask.npy", result['mask'])
        np.save(case_output_dir / "labeled_mask.npy", result['labeled_mask'])
        np.save(case_output_dir / "trajectories.npy", np.array(result['trajectories'], dtype=object))
        np.save(case_output_dir / "centers.npy", result['centers'])
        np.save(case_output_dir / "widths.npy", result['widths'])
        
        results.append({
            'case_name': case_meta['name'],
            'output_dir': str(case_output_dir),
        })
    
    # Save results metadata
    results_metadata_path = output_dir / "results_metadata.json"
    with open(results_metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Stage 4 complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
