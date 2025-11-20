"""Stage 1: Generate synthetic test cases for the demo."""

import numpy as np
from pathlib import Path
import sys
import json

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from kymo_tracker.utils.helpers import (
    generate_kymograph,
    get_diffusion_coefficient,
)


def generate_demo_cases(output_dir: Path):
    """Generate 5 test cases with different scenarios."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    
    # Case 1: Single particle, low noise
    print("Generating case 1: Single particle, low noise...")
    radius = 10.0
    diffusion = get_diffusion_coefficient(radius)
    noisy, gt, paths = generate_kymograph(
        length=2000, width=512,
        diffusion=diffusion,
        contrast=0.8,
        noise_level=0.15,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=42,
    )
    if paths.ndim == 1:
        true_paths_list = [paths]
    elif paths.ndim == 2:
        true_paths_list = [paths[i] for i in range(paths.shape[0])]
    else:
        true_paths_list = []
    
    case1_path = output_dir / "case_01.npy"
    np.save(case1_path, noisy)
    cases.append({
        'name': 'Single Particle (Low Noise)',
        'noisy_path': str(case1_path),
        'true_paths': true_paths_list,
    })
    
    # Case 2: Single particle, high noise
    print("Generating case 2: Single particle, high noise...")
    radius = 15.0
    diffusion = get_diffusion_coefficient(radius)
    noisy, gt, paths = generate_kymograph(
        length=2000, width=512,
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
    
    case2_path = output_dir / "case_02.npy"
    np.save(case2_path, noisy)
    cases.append({
        'name': 'Single Particle (High Noise)',
        'noisy_path': str(case2_path),
        'true_paths': true_paths_list,
    })
    
    # Case 3: Two particles, moderate noise
    print("Generating case 3: Two particles, moderate noise...")
    radius1, radius2 = 8.0, 20.0
    diffusion1 = get_diffusion_coefficient(radius1)
    diffusion2 = get_diffusion_coefficient(radius2)
    noisy, gt, paths = generate_kymograph(
        length=2000, width=512,
        diffusion=[diffusion1, diffusion2],
        contrast=[0.7, 0.5],
        noise_level=0.3,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=44,
    )
    case3_path = output_dir / "case_03.npy"
    np.save(case3_path, noisy)
    cases.append({
        'name': 'Two Particles (Moderate Noise)',
        'noisy_path': str(case3_path),
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 4: Three particles, moderate noise
    print("Generating case 4: Three particles, moderate noise...")
    radii = [5.0, 12.0, 25.0]
    diffusions = [get_diffusion_coefficient(r) for r in radii]
    noisy, gt, paths = generate_kymograph(
        length=2000, width=512,
        diffusion=diffusions,
        contrast=[0.8, 0.6, 0.5],
        noise_level=0.25,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=45,
    )
    case4_path = output_dir / "case_04.npy"
    np.save(case4_path, noisy)
    cases.append({
        'name': 'Three Particles (Moderate Noise)',
        'noisy_path': str(case4_path),
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Case 5: Two particles, high noise
    print("Generating case 5: Two particles, high noise...")
    radius1, radius2 = 10.0, 18.0
    diffusion1 = get_diffusion_coefficient(radius1)
    diffusion2 = get_diffusion_coefficient(radius2)
    noisy, gt, paths = generate_kymograph(
        length=2000, width=512,
        diffusion=[diffusion1, diffusion2],
        contrast=[0.5, 0.4],
        noise_level=0.5,
        peak_width=1.0,
        dx=0.5, dt=1.0,
        seed=46,
    )
    case5_path = output_dir / "case_05.npy"
    np.save(case5_path, noisy)
    cases.append({
        'name': 'Two Particles (High Noise)',
        'noisy_path': str(case5_path),
        'true_paths': [paths[i] for i in range(paths.shape[0])],
    })
    
    # Save metadata
    metadata_path = output_dir / "cases_metadata.json"
    # Convert numpy arrays to lists for JSON serialization
    metadata = []
    for case in cases:
        metadata.append({
            'name': case['name'],
            'noisy_path': case['noisy_path'],
            'true_paths': [path.tolist() if isinstance(path, np.ndarray) else path for path in case['true_paths']],
        })
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {len(cases)} test cases")
    print(f"Data saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    return cases


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic test cases")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo/data",
        help="Directory to save generated test cases",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STAGE 1: Generating Test Cases")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    generate_demo_cases(output_dir)
    
    print("\n" + "=" * 70)
    print("Stage 1 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
