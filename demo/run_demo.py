"""Main demo script that runs training and inference, then creates comparison plots."""

import numpy as np
from pathlib import Path
import sys
import os

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from kymo_tracker.utils.helpers import (
    generate_kymograph,
    get_diffusion_coefficient,
    find_max_subpixel,
)
from kymo_tracker.classical.pipeline import classical_median_threshold_tracking
from kymo_tracker.deeplearning.training.multitask import (
    MultiTaskConfig,
    train_multitask_model,
    save_multitask_model,
    load_multitask_model,
)
from kymo_tracker.data.multitask_dataset import MultiTaskDataset
from kymo_tracker.deeplearning.predict import (
    process_slice_independently,
    link_trajectories_across_slices,
)
from kymo_tracker.utils.device import get_default_device
from kymo_tracker.utils.visualization import visualize_comparison


def run_classical_pipeline(kymograph_noisy):
    """Run classical median filter + thresholding pipeline."""
    result = classical_median_threshold_tracking(
        kymograph_noisy,
        median_kernel=(11, 11),
        threshold_mode="otsu",
        min_component_size=50,  # Increased from 8 to filter out small noise regions
    )
    
    # Extract trajectories
    trajectories = []
    if result.trajectories:
        for traj in result.trajectories:
            trajectories.append(traj)
    # Ensure at least one empty trajectory if none found
    if not trajectories:
        trajectories.append(np.full(len(kymograph_noisy), np.nan))
    
    # Create combined mask from all instance masks
    combined_mask = np.zeros_like(kymograph_noisy, dtype=bool)
    for instance_mask in result.instance_masks:
        combined_mask |= instance_mask
    
    return {
        'filtered': result.filtered,
        'mask': combined_mask,
        'labeled_mask': result.labeled_mask,
        'trajectories': trajectories,
    }


def run_deeplearning_pipeline(kymograph_noisy, model, device):
    """Run deep learning denoising + locator pipeline."""
    T, W = kymograph_noisy.shape
    chunk_size = 16
    overlap = 8
    
    # Process each slice independently
    slice_results = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        slice_data = kymograph_noisy[start:end]
        slice_result = process_slice_independently(model, slice_data, device=device)
        slice_results.append(slice_result)
        start += chunk_size - overlap
    
    # Link trajectories
    slice_trajectories_list = [result['trajectories'] for result in slice_results]
    linked_trajectories = link_trajectories_across_slices(
        slice_trajectories_list,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    
    # Reconstruct full outputs for visualization
    denoised_full = np.zeros((T, W), dtype=np.float32)
    weights = np.zeros((T, W), dtype=np.float32)
    mask_full = np.zeros((T, W), dtype=bool)
    labeled_mask_full = np.zeros((T, W), dtype=int)
    
    window = np.ones(chunk_size)
    if overlap > 0:
        fade_len = overlap // 2
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)
    
    centers_full = None
    widths_full = None
    temporal_weights = np.zeros((T, 1), dtype=np.float32)
    
    start = 0
    for result in slice_results:
        end = min(start + chunk_size, T)
        actual_len = end - start
        
        weight_chunk = window[:actual_len, np.newaxis]
        denoised_full[start:end] += result['denoised'] * weight_chunk
        weights[start:end] += weight_chunk
        mask_full[start:end] = result['mask']
        labeled_mask_full[start:end] = result['labeled_mask']
        
        if centers_full is None:
            n_tracks = result['centers'].shape[1] if result['centers'].ndim > 1 else 1
            centers_full = np.zeros((T, n_tracks), dtype=np.float32)
            widths_full = np.zeros((T, n_tracks), dtype=np.float32)
        
        centers_full[start:end] += result['centers'] * weight_chunk
        widths_full[start:end] += result['widths'] * weight_chunk
        temporal_weights[start:end] += weight_chunk
        
        start += chunk_size - overlap
    
    denoised_full = np.divide(denoised_full, weights, out=np.zeros_like(denoised_full), where=weights > 0)
    centers_full = np.divide(centers_full, temporal_weights, out=np.zeros_like(centers_full), where=temporal_weights > 0)
    widths_full = np.divide(widths_full, temporal_weights, out=np.zeros_like(widths_full), where=temporal_weights > 0)
    
    return {
        'denoised': denoised_full,
        'mask': mask_full,
        'labeled_mask': labeled_mask_full,
        'trajectories': linked_trajectories,
        'centers': centers_full,
        'widths': widths_full,
    }


def generate_demo_cases():
    """Generate 5 test cases with different scenarios."""
    cases = []
    
    # Case 1: Single particle, low noise
    print("Generating case 1: Single particle, low noise...")
    radius = 10.0
    diffusion = get_diffusion_coefficient(radius)
    noisy, gt, paths = generate_kymograph(
        length=512, width=512,
        diffusion=diffusion,
        contrast=0.8,
        noise_level=0.15,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=42,
    )
    # Handle paths: always 2D (n_particles, length), convert to list
    if paths.ndim == 1:
        true_paths_list = [paths]
    elif paths.ndim == 2:
        true_paths_list = [paths[i] for i in range(paths.shape[0])]
    else:
        true_paths_list = []
    cases.append({
        'name': 'Single Particle (Low Noise)',
        'noisy': noisy,
        'true_paths': true_paths_list,
    })
    
    # Case 2: Single particle, high noise
    print("Generating case 2: Single particle, high noise...")
    radius = 15.0
    diffusion = get_diffusion_coefficient(radius)
    noisy, gt, paths = generate_kymograph(
        length=512, width=512,
        diffusion=diffusion,
        contrast=0.6,
        noise_level=0.4,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=43,
    )
    if paths.ndim == 1:
        true_paths_list = [paths]
    elif paths.ndim == 2:
        true_paths_list = [paths[i] for i in range(paths.shape[0])]
    else:
        true_paths_list = []
    cases.append({
        'name': 'Single Particle (High Noise)',
        'noisy': noisy,
        'true_paths': true_paths_list,
    })
    
    # Case 3: Two particles, moderate noise
    print("Generating case 3: Two particles, moderate noise...")
    radius1, radius2 = 8.0, 20.0
    diffusion1 = get_diffusion_coefficient(radius1)
    diffusion2 = get_diffusion_coefficient(radius2)
    noisy, gt, paths = generate_kymograph(
        length=512, width=512,
        diffusion=[diffusion1, diffusion2],
        contrast=[0.7, 0.5],
        noise_level=0.3,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=44,
    )
    cases.append({
        'name': 'Two Particles (Moderate Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 4: Three particles, moderate noise
    print("Generating case 4: Three particles, moderate noise...")
    radii = [5.0, 12.0, 25.0]
    diffusions = [get_diffusion_coefficient(r) for r in radii]
    noisy, gt, paths = generate_kymograph(
        length=512, width=512,
        diffusion=diffusions,
        contrast=[0.8, 0.6, 0.5],
        noise_level=0.25,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=45,
    )
    cases.append({
        'name': 'Three Particles (Moderate Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 5: Two particles, high noise
    print("Generating case 5: Two particles, high noise...")
    radius1, radius2 = 10.0, 18.0
    diffusion1 = get_diffusion_coefficient(radius1)
    diffusion2 = get_diffusion_coefficient(radius2)
    noisy, gt, paths = generate_kymograph(
        length=512, width=512,
        diffusion=[diffusion1, diffusion2],
        contrast=[0.5, 0.4],
        noise_level=0.5,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=46,
    )
    cases.append({
        'name': 'Two Particles (High Noise)',
        'noisy': noisy,
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    return cases


def main():
    """Main demo function."""
    print("=" * 70)
    print("KYMO-TRACKER DEMO: Classical vs Deep Learning Comparison")
    print("=" * 70)
    
    # Setup paths
    demo_dir = Path(__file__).parent
    model_path = demo_dir.parent / "artifacts" / "demo_model.pth"
    checkpoint_dir = demo_dir.parent / "checkpoints"
    output_dir = demo_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Step 1: Train model if it doesn't exist
    if not model_path.exists():
        print("\n" + "=" * 70)
        print("STEP 1: Training model (this may take a while)...")
        print("=" * 70)
        
        # Lightweight training for demo
        dataset = MultiTaskDataset(
            n_samples=1024,  # Reduced for faster training
            window_length=16,
            length=512,
            width=512,
            radii_nm=(3.0, 70.0),
            contrast=(0.5, 1.1),
            noise_level=(0.08, 0.8),
            multi_trajectory_prob=1.0,
            max_trajectories=3,
            mask_peak_width_samples=10.0,
        )
        
        config = MultiTaskConfig(
            epochs=15,
            batch_size=16,
            learning_rate=1.5e-3,
            checkpoint_dir=str(checkpoint_dir),
            auto_resume=False,  # Don't resume for demo
        )
        
        model = train_multitask_model(config, dataset)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_multitask_model(model, str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print(f"\nModel found at {model_path}, skipping training...")
    
    # Step 2: Load model
    print("\n" + "=" * 70)
    print("STEP 2: Loading model...")
    print("=" * 70)
    model = load_multitask_model(str(model_path), device=device, max_tracks=3)
    print("Model loaded successfully")
    
    # Step 3: Generate test cases
    print("\n" + "=" * 70)
    print("STEP 3: Generating test cases...")
    print("=" * 70)
    test_cases = generate_demo_cases()
    
    # Step 4: Run inference and create plots
    print("\n" + "=" * 70)
    print("STEP 4: Running inference and creating comparison plots...")
    print("=" * 70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nProcessing case {i}/5: {case['name']}...")
        
        noisy_kymo = case['noisy']
        true_paths = case.get('true_paths', None)
        
        # Run classical pipeline
        print("  Running classical pipeline...")
        classical_result = run_classical_pipeline(noisy_kymo)
        
        # Run deep learning pipeline
        print("  Running deep learning pipeline...")
        dl_result = run_deeplearning_pipeline(noisy_kymo, model, device)
        
        # Create comparison plot
        print("  Creating comparison plot...")
        output_path = output_dir / f"comparison_case_{i:02d}_{case['name'].replace(' ', '_').lower()}.png"
        visualize_comparison(
            noisy_kymo=noisy_kymo,
            classical_filtered=classical_result['filtered'],
            classical_mask=classical_result['mask'],
            classical_trajectories=classical_result['trajectories'],
            deeplearning_denoised=dl_result['denoised'],
            deeplearning_mask=dl_result['mask'],
            deeplearning_trajectories=dl_result['trajectories'],
            true_paths=true_paths,
            output_path=str(output_path),
            title=f"Case {i}: {case['name']}",
        )
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"All comparison plots saved to: {output_dir}")
    print(f"Generated {len(test_cases)} comparison plots")


if __name__ == "__main__":
    main()
