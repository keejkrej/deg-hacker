"""Stage 3: Run classical inference pipeline on test cases."""

import numpy as np
from pathlib import Path
import sys
import json

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from kymo_tracker.classical.pipeline import classical_median_threshold_tracking


def run_classical_pipeline(kymograph_noisy):
    """Run classical median filter + thresholding pipeline."""
    result = classical_median_threshold_tracking(
        kymograph_noisy,
        median_kernel=(11, 11),
        threshold_mode="otsu",
        min_component_size=8,
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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run classical inference pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="demo/data",
        help="Directory containing test cases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo/results/classical",
        help="Directory to save classical results",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STAGE 3: Running Classical Pipeline")
    print("=" * 70)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Run classical pipeline
        print("  Running classical pipeline...")
        result = run_classical_pipeline(noisy_kymo)
        
        # Save results
        case_output_dir = output_dir / f"case_{i:02d}"
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(case_output_dir / "filtered.npy", result['filtered'])
        np.save(case_output_dir / "mask.npy", result['mask'])
        np.save(case_output_dir / "labeled_mask.npy", result['labeled_mask'])
        np.save(case_output_dir / "trajectories.npy", np.array(result['trajectories'], dtype=object))
        
        results.append({
            'case_name': case_meta['name'],
            'output_dir': str(case_output_dir),
        })
    
    # Save results metadata
    results_metadata_path = output_dir / "results_metadata.json"
    with open(results_metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Stage 3 complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
