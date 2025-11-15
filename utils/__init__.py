"""
Utils Package

Provides analysis utilities, tracking functions, helper functions, and data structures.
"""

# Import everything from analysis.py for backward compatibility
from utils.analysis import (
    SimulationData,
    AnalysisMetrics,
    MultiSimulationData,
    METRIC_FIELDNAMES,
    simulate_single_particle,
    simulate_multi_particle,
    estimate_noise_and_contrast,
    write_joint_metrics_csv,
    summarize_analysis,
)

# Import tracking functions
from utils.tracking import (
    TrackSummary,
    track_particles,
    analyze_multi_particle,
    summarize_multi_particle_analysis,
    run_parameter_grid,
)

# Import helper functions (commonly used ones)
from utils.helpers import (
    find_max_subpixel,
    get_diffusion_coefficient,
    get_particle_radius,
    estimate_diffusion_msd_fit,
    generate_kymograph,
    load_challenge_data,
    load_challenge_data_multiple_particles,
)

__all__ = [
    # Data structures
    "SimulationData",
    "AnalysisMetrics",
    "MultiSimulationData",
    "TrackSummary",
    "METRIC_FIELDNAMES",
    # Simulation functions
    "simulate_single_particle",
    "simulate_multi_particle",
    # Analysis functions
    "estimate_noise_and_contrast",
    "write_joint_metrics_csv",
    "summarize_analysis",
    # Tracking functions
    "track_particles",
    "analyze_multi_particle",
    "summarize_multi_particle_analysis",
    "run_parameter_grid",
    # Helper functions
    "find_max_subpixel",
    "get_diffusion_coefficient",
    "get_particle_radius",
    "estimate_diffusion_msd_fit",
    "generate_kymograph",
    "load_challenge_data",
    "load_challenge_data_multiple_particles",
]
