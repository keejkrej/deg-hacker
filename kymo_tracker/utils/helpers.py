import numpy as np
import matplotlib.pyplot as plt


def get_diffusion_coefficient(particle_radius, temperature_deg=25, viscosity=0.89e-3):
    """Calculate diffusion coefficient in um^2/ms from particle radius using Stokes-Einstein equation.
    Parameters:
        particle_radius (float): Particle radius in nanometers.
    """
    particle_radius_m = particle_radius * 1e-9
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    temperature_K = temperature_deg + 273.15  # Convert to Kelvin
    D = k_B * temperature_K / (6 * np.pi * viscosity * particle_radius_m) * 1e12 / 1e3  # Convert to um^2/ms
    return D


def get_particle_radius(diffusion_coefficient, temperature_deg=25, viscosity=0.89e-3):
    """Calculate particle radius in nanometers from diffusion coefficient using Stokes-Einstein equation.

    Parameters:
        diffusion_coefficient (float): Diffusion coefficient in um^2/ms.
    """
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    temperature_K = temperature_deg + 273.15  # Convert to Kelvin
    particle_radius_m = k_B * temperature_K / (6 * np.pi * viscosity * diffusion_coefficient * 1e3 / 1e12)
    particle_radius_nm = particle_radius_m * 1e9
    return particle_radius_nm


def estimate_diffusion_msd_fit(positions_um, dt=1, dx = 0.5, max_lag=10):
    """Fit MSD vs time to extract true diffusion. Using a MSD fit reduces bias from smoothing of trajectories.
    
    Parameters:
        positions_um (array-like): Particle positions in micrometers.
        max_lag (int): Maximum lag time for MSD calculation.
        dt_ms (float): Time step between positions in milliseconds.
        dx_um (float): Spatial step between positions in micrometers.
    
    Returns:
        float: Diffusion coefficient in um^2/ms.
    """
    lags = np.arange(1, max_lag+1)
    msds = []
    for lag in lags:
        displacements = positions_um[lag:] - positions_um[:-lag]
        displacements = displacements[~np.isnan(displacements)]
        displacements = displacements[displacements != 0]
        msd = np.mean(displacements**2)
        msds.append(msd)

    # Convert dt_ms to seconds for the calculation
    dt = dt / 1000.0
    # Fit MSD = 2*D*dt*lag (slope = 2*D*dt)
    slope, _ = np.polyfit(lags * dt, msds, 1)
    D_um2_s = slope / 2

    # Convert diffusion coefficient from micrometers^2/s to um^2/ms
    D_um2_ms = D_um2_s / 1000.0 * dx**2
    return D_um2_ms


def generate_kymograph(length=400, width=128, contrast=1.0, diffusion=1.0,
                       noise_level=1.0, peak_width=1.0, dt=1.0, dx=0.5,
                       seed=None):
    """
    Multi-particle Brownian kymograph.

    contrast: float or list/array of floats (one per particle)
    diffusion: float or list/array of floats (um^2/ms, one per particle)
    peak_width: global Gaussian width (micrometers)
    dt: time step (ms)
    dx: spatial sampling (micrometers)

    Returns:
        noisy : (length, width)
        gt    : (length, width) sum of particle Gaussians
        paths : (n_particles, length) particle positions (in samples)
    """
    if seed is not None:
        np.random.seed(seed)

    # Normalize inputs to lists
    if np.isscalar(contrast):
        contrasts = [float(contrast)]
    else:
        contrasts = list(contrast)

    if np.isscalar(diffusion):
        diffusions = [float(diffusion)]
    else:
        diffusions = list(diffusion)

    if len(contrasts) != len(diffusions):
        raise ValueError("contrast and diffusion must have same length when lists are provided.")

    n_particles = len(contrasts)
    # Initial positions: distribute evenly across the width
    positions = np.linspace(0, width - 1, n_particles+1, endpoint=False).astype(float)[1:]
    

    gt = np.zeros((length, width), dtype=float)
    paths = np.zeros((n_particles, length), dtype=float)
    xs = np.arange(width, dtype=float)
    w_samples = peak_width / dx  # global width in samples

    step_sigmas = [np.sqrt(2 * d * dt) / dx for d in diffusions]

    for t in range(length):
        row = np.zeros(width, dtype=float)
        for i in range(n_particles):
            # Brownian step
            positions[i] += np.random.normal(0, step_sigmas[i])
            # Reflect boundaries
            if positions[i] < 0:
                positions[i] = -positions[i]
            if positions[i] > width - 1:
                positions[i] = 2 * (width - 1) - positions[i]
            paths[i, t] = positions[i]
            row += contrasts[i] * np.exp(-0.5 * ((xs - positions[i]) / w_samples) ** 2)
        gt[t] = row

    noisy = np.clip(gt + noise_level * np.random.randn(length, width), 0, 1)
    return noisy, gt, paths


def find_max_subpixel(I):
    """Find subpixel maximum positions in each row of input 2D array I using parabolic interpolation."""
    positions = []
    for row in I:
        max_idx = np.argmax(row)
        # Exclude extreme positions (first or last column)
        if max_idx == 0 or max_idx == len(row) - 1:
            positions.append(np.nan)
            continue
        y0, y1, y2 = row[max_idx - 1], row[max_idx], row[max_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom == 0:
            positions.append(max_idx)
        else:
            delta = 0.5 * (y0 - y2) / denom
            positions.append(max_idx + delta)
    return np.array(positions)


def load_challenge_data(index):
    filename = f"data/kymograph_noisy_{index}.npy"
    data = np.load(filename)
    return data

def load_challenge_data_multiple_particles(index):
    filename = f"data/kymograph_noisy_multiple_particles_{index}.npy"
    data = np.load(filename)
    return data