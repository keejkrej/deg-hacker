from helpers import (
    find_max_subpixel,
    get_diffusion_coefficient,
    get_particle_radius,
    estimate_diffusion_msd_fit,
    generate_kymograph,
)

from scipy.signal import medfilt

particle_size = 5  # in nm
diffusion_coefficient = get_diffusion_coefficient(particle_size)
x_step = 0.5  # each sample is 0.5µm. This is determined by the microscope pixel size and magnification
t_step = 1.0  # each time step is 1ms. This is determined by the camera frame rate
kymograph_noisy, kymograph_gt, true_path = generate_kymograph(
    length=4000,
    width=256,
    diffusion=diffusion_coefficient,
    contrast=1,
    noise_level=0.3,
    peak_width=1,
    dx=x_step,
    dt=t_step,
)

kymograph_noisy_filtered = medfilt(kymograph_noisy, kernel_size=(5, 1))
estimated_path = find_max_subpixel(kymograph_noisy)
estimated_path_filtered = find_max_subpixel(kymograph_noisy_filtered)

estimated_diffusion = estimate_diffusion_msd_fit(estimated_path, dx=x_step, dt=t_step)
estimated_diffusion_filtered = estimate_diffusion_msd_fit(
    estimated_path_filtered, dx=x_step, dt=t_step
)

estimated_radius = get_particle_radius(estimated_diffusion)
estimated_radius_filtered = get_particle_radius(estimated_diffusion_filtered)

print(f"True Diffusion Coefficient: {diffusion_coefficient:.4f} µm²/ms")
print(f"Estimated Diffusion Coefficient: {estimated_diffusion:.4f} µm²/ms")
print(
    f"Estimated Diffusion Coefficient Filtered: {estimated_diffusion_filtered:.4f} µm²/ms"
)
print(f"True Particle Radius: {particle_size:.3f} nm")
print(f"Estimated Particle Radius: {estimated_radius:.3f} nm")
print(f"Estimated Particle Radius Filtered: {estimated_radius_filtered:.3f} nm")
