"""Stage 5: Create comparison visualization plots."""

import numpy as np
from pathlib import Path
import sys
import json

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from kymo_tracker.utils.visualization import visualize_comparison


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create comparison visualization plots")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="demo/data",
        help="Directory containing test cases",
    )
    parser.add_argument(
        "--classical-dir",
        type=str,
        default="demo/results/classical",
        help="Directory containing classical results",
    )
    parser.add_argument(
        "--deeplearning-dir",
        type=str,
        default="demo/results/deeplearning",
        help="Directory containing deep learning results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo/results",
        help="Directory to save comparison plots",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STAGE 5: Creating Comparison Plots")
    print("=" * 70)
    
    data_dir = Path(args.data_dir)
    classical_dir = Path(args.classical_dir)
    deeplearning_dir = Path(args.deeplearning_dir)
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
    for i, case_meta in enumerate(cases_metadata, 1):
        print(f"\nCreating plot for case {i}/{len(cases_metadata)}: {case_meta['name']}...")
        
        # Load noisy kymograph
        noisy_path = Path(case_meta['noisy_path'])
        noisy_kymo = np.load(noisy_path)
        
        # Load true paths
        true_paths = None
        if 'true_paths' in case_meta and case_meta['true_paths']:
            true_paths = [np.array(path) for path in case_meta['true_paths']]
        
        # Load classical results
        classical_case_dir = classical_dir / f"case_{i:02d}"
        classical_filtered = np.load(classical_case_dir / "filtered.npy")
        classical_mask = np.load(classical_case_dir / "mask.npy")
        classical_trajectories_obj = np.load(classical_case_dir / "trajectories.npy", allow_pickle=True)
        # Handle both list and array formats
        if isinstance(classical_trajectories_obj, np.ndarray) and classical_trajectories_obj.ndim == 2:
            classical_trajectories = [classical_trajectories_obj[i] for i in range(classical_trajectories_obj.shape[0])]
        else:
            classical_trajectories = [traj for traj in classical_trajectories_obj]
        
        # Load deep learning results
        dl_case_dir = deeplearning_dir / f"case_{i:02d}"
        dl_denoised = np.load(dl_case_dir / "denoised.npy")
        dl_heatmap = np.load(dl_case_dir / "heatmap.npy")
        dl_trajectories_obj = np.load(dl_case_dir / "trajectories.npy", allow_pickle=True)
        # Handle both list and array formats
        if isinstance(dl_trajectories_obj, np.ndarray) and dl_trajectories_obj.ndim == 2:
            dl_trajectories = [dl_trajectories_obj[i] for i in range(dl_trajectories_obj.shape[0])]
        else:
            dl_trajectories = [traj for traj in dl_trajectories_obj]
        
        # Create comparison plot
        print("  Creating comparison plot...")
        output_filename = f"comparison_case_{i:02d}_{case_meta['name'].replace(' ', '_').lower()}.png"
        output_path = output_dir / output_filename
        
        visualize_comparison(
            noisy_kymo=noisy_kymo,
            classical_filtered=classical_filtered,
            classical_mask=classical_mask,
            classical_trajectories=classical_trajectories,
            deeplearning_denoised=dl_denoised,
            deeplearning_heatmap=dl_heatmap,
            deeplearning_trajectories=dl_trajectories,
            true_paths=true_paths,
            output_path=str(output_path),
            title=f"Case {i}: {case_meta['name']}",
        )
        
        print(f"  Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("Stage 5 complete!")
    print(f"All comparison plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
