"""
Utils Package

Provides analysis utilities, tracking functions, and data structures.
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
]
