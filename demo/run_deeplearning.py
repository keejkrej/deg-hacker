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
    process_slice_independently,
    link_trajectories_across_slices,
)
from kymo_tracker.deeplearning.training.multitask import load_multitask_model
from kymo_tracker.utils.device import get_default_device


def run_deeplearning_pipeline(kymograph_noisy, model, device):
    """
    Run deep learning denoising + locator pipeline.
    
    Processes each 16x512 slice independently, extracts trajectories per slice,
    then links trajectories across slices at the end.
    """
    T, W = kymograph_noisy.shape
    chunk_size = 16
    overlap = 8
    
    # Process each slice independently
    slice_results = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        slice_data = kymograph_noisy[start:end]
        
        # Process this slice independently
        slice_result = process_slice_independently(
            model, slice_data, device=device
        )
        slice_results.append(slice_result)
        
        start += chunk_size - overlap
    
    # Collect trajectories from each slice
    slice_trajectories_list = [result['trajectories'] for result in slice_results]
    
    # Link trajectories across slices
    linked_trajectories = link_trajectories_across_slices(
        slice_trajectories_list,
        chunk_size=chunk_size,
        overlap=overlap,
        total_length=T,  # Pass actual kymograph length
    )
    
    # Reconstruct full denoised kymograph and masks (for visualization)
    # Blend denoised slices with overlap handling
    denoised_full = np.zeros((T, W), dtype=np.float32)
    weights = np.zeros((T, W), dtype=np.float32)
    mask_full = np.zeros((T, W), dtype=bool)
    labeled_mask_full = np.zeros((T, W), dtype=int)
    
    window = np.ones(chunk_size)
    if overlap > 0:
        fade_len = overlap // 2
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)
    
    start = 0
    for i, result in enumerate(slice_results):
        end = min(start + chunk_size, T)
        actual_len = end - start
        
        weight_chunk = window[:actual_len, np.newaxis]
        denoised_full[start:end] += result['denoised'] * weight_chunk
        weights[start:end] += weight_chunk
        
        # For masks, just take the last slice's mask in overlap regions
        mask_full[start:end] = result['mask']
        labeled_mask_full[start:end] = result['labeled_mask']
        
        start += chunk_size - overlap
    
    denoised_full = np.divide(denoised_full, weights, out=np.zeros_like(denoised_full), where=weights > 0)
    
    # Reconstruct heatmap (blended across slices)
    heatmap_full = np.zeros((T, W), dtype=np.float32)
    weights_heatmap = np.zeros((T, W), dtype=np.float32)
    
    start = 0
    for i, result in enumerate(slice_results):
        end = min(start + chunk_size, T)
        actual_len = end - start
        
        weight_chunk = window[:actual_len, np.newaxis]
        heatmap_full[start:end] += result['heatmap'] * weight_chunk
        weights_heatmap[start:end] += weight_chunk
        
        start += chunk_size - overlap
    
    heatmap_full = np.divide(heatmap_full, weights_heatmap, out=np.zeros_like(heatmap_full), where=weights_heatmap > 0)
    
    # Derive centers/widths from trajectories for backward compatibility
    n_tracks = len(linked_trajectories)
    centers_full = np.full((T, n_tracks), np.nan, dtype=np.float32)
    widths_full = np.full((T, n_tracks), np.nan, dtype=np.float32)
    
    for track_idx, traj in enumerate(linked_trajectories):
        # Trim or pad trajectory to match kymograph length
        if len(traj) > T:
            traj_trimmed = traj[:T]
        elif len(traj) < T:
            # Pad with NaN if trajectory is shorter
            traj_trimmed = np.full(T, np.nan, dtype=traj.dtype)
            traj_trimmed[:len(traj)] = traj
        else:
            traj_trimmed = traj
        centers_full[:, track_idx] = traj_trimmed
        # Estimate width from heatmap (FWHM around peak)
        for t in range(T):
            if not np.isnan(traj_trimmed[t]):
                center_px = int(np.round(traj_trimmed[t]))
                center_px = np.clip(center_px, 0, W - 1)
                # Find width at half maximum
                peak_value = heatmap_full[t, center_px]
                if peak_value > 0:
                    threshold = peak_value * 0.5
                    # Find left and right boundaries
                    left_idx = center_px
                    while left_idx > 0 and heatmap_full[t, left_idx] > threshold:
                        left_idx -= 1
                    right_idx = center_px
                    while right_idx < W - 1 and heatmap_full[t, right_idx] > threshold:
                        right_idx += 1
                    width_px = right_idx - left_idx
                    if width_px > 0:
                        widths_full[t, track_idx] = width_px
    
    return {
        'denoised': denoised_full,
        'mask': mask_full,
        'labeled_mask': labeled_mask_full,
        'trajectories': linked_trajectories,
        'heatmap': heatmap_full,
        'centers': centers_full,
        'widths': widths_full,
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
        np.save(case_output_dir / "heatmap.npy", result['heatmap'])
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
