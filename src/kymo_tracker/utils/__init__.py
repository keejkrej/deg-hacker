"""
Utils Package

Provides analysis utilities, tracking functions, helper functions, and data structures.
"""

# Import everything from analysis.py for backward compatibility
from kymo_tracker.utils.analysis import (
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

# Import tracking functions (optional - requires sklearn)
try:
    from kymo_tracker.utils.tracking import (
        TrackSummary,
        track_particles,
        analyze_multi_particle,
        summarize_multi_particle_analysis,
        run_parameter_grid,
    )
    _tracking_available = True
except ImportError as e:
    # Tracking functions require sklearn - make them optional
    _tracking_available = False
    TrackSummary = None
    track_particles = None
    analyze_multi_particle = None
    summarize_multi_particle_analysis = None
    run_parameter_grid = None

# Import helper functions (commonly used ones)
from kymo_tracker.utils.helpers import (
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
    "METRIC_FIELDNAMES",
    # Simulation functions
    "simulate_single_particle",
    "simulate_multi_particle",
    # Analysis functions
    "estimate_noise_and_contrast",
    "write_joint_metrics_csv",
    "summarize_analysis",
    # Helper functions
    "find_max_subpixel",
    "get_diffusion_coefficient",
    "get_particle_radius",
    "estimate_diffusion_msd_fit",
    "generate_kymograph",
    "load_challenge_data",
    "load_challenge_data_multiple_particles",
]

# Add tracking functions to __all__ only if available
if _tracking_available:
    __all__.extend([
        "TrackSummary",
        "track_particles",
        "analyze_multi_particle",
        "summarize_multi_particle_analysis",
        "run_parameter_grid",
    ])

from kymo_tracker.utils.device import get_default_device

__all__.append("get_default_device")
