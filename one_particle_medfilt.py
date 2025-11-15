from helpers import (
    find_max_subpixel,
    get_diffusion_coefficient,
    get_particle_radius,
    estimate_diffusion_msd_fit,
    generate_kymograph,
)

from scipy.signal import medfilt
import matplotlib.pyplot as plt
import os


def analyze_particle(p, c, n):
    """
    Analyze particle diffusion from kymograph data.

    Parameters:
    -----------
    p : float
        Particle size in nm
    c : float
        Contrast
    n : float
        Noise level
    """
    d = get_diffusion_coefficient(p)
    x_step = 0.5
    t_step = 1.0
    n_t = 4000
    n_x = 256
    kymograph_noisy, kymograph_gt, true_path = generate_kymograph(
        length=n_t,
        width=n_x,
        diffusion=d,
        contrast=c,
        noise_level=n,
        peak_width=1,
        dx=x_step,
        dt=t_step,
    )

    kymograph_noisy_filtered = medfilt(kymograph_noisy, kernel_size=(5, 3))
    estimated_path = find_max_subpixel(kymograph_noisy)
    estimated_path_filtered = find_max_subpixel(kymograph_noisy_filtered)

    estimated_diffusion_true = estimate_diffusion_msd_fit(
        true_path.T, dx=x_step, dt=t_step
    )
    estimated_diffusion = estimate_diffusion_msd_fit(
        estimated_path, dx=x_step, dt=t_step
    )
    estimated_diffusion_filtered = estimate_diffusion_msd_fit(
        estimated_path_filtered, dx=x_step, dt=t_step
    )
    estimated_radius_true = get_particle_radius(estimated_diffusion_true)
    estimated_radius = get_particle_radius(estimated_diffusion)
    estimated_radius_filtered = get_particle_radius(estimated_diffusion_filtered)

    print(f"True Diffusion Coefficient: {d:.4f} µm²/ms")
    print(
        f"Estimated Diffusion Coefficient True Path: {estimated_diffusion_true:.4f} µm²/ms"
    )
    print(f"Estimated Diffusion Coefficient: {estimated_diffusion:.4f} µm²/ms")
    print(
        f"Estimated Diffusion Coefficient Filtered: {estimated_diffusion_filtered:.4f} µm²/ms"
    )
    print(f"True Particle Radius: {p:.3f} nm")
    print(f"Estimated Particle Radius True Path: {estimated_radius_true:.3f} nm")
    print(f"Estimated Particle Radius: {estimated_radius:.3f} nm")
    print(f"Estimated Particle Radius Filtered: {estimated_radius_filtered:.3f} nm")

    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    fig.suptitle(
        f"p={p} nm, c={c}, n={n}, D={d:.4f} µm²/ms \n D_estimated_true={estimated_diffusion_true:.4f} µm²/ms, D_estimated={estimated_diffusion:.4f} µm²/ms, D_estimated_filtered={estimated_diffusion_filtered:.4f} µm²/ms"
    )
    ax[0, 0].imshow(
        kymograph_noisy.T,
        aspect="auto",
        origin="lower",
        extent=[0, n_t, 0, n_x],
        vmin=0,
        vmax=0.5,
    )
    ax[0, 1].plot(true_path.T)
    ax[0, 1].set_xlim(0, n_t)
    ax[0, 1].set_ylim(0, n_x)
    ax[1, 0].plot(estimated_path)
    ax[1, 0].set_xlim(0, n_t)
    ax[1, 0].set_ylim(0, n_x)
    ax[1, 1].plot(estimated_path_filtered)
    ax[1, 1].set_xlim(0, n_t)
    ax[1, 1].set_ylim(0, n_x)
    ax[0, 0].set_title("Noisy Kymograph")
    ax[0, 1].set_title("True Path")
    ax[1, 0].set_title("Dumbest Approach")
    ax[1, 1].set_title("Second Dumbest Approach")

    fig_filename = f"figures/medfilt/one_particle_p_{p}_c_{c}_n_{n}.png"
    fig.savefig(fig_filename)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/medfilt", exist_ok=True)
    # for p in [2.5, 5, 10]:
    #     for c in [0.6, 0.8, 1.0]:
    #         for n in [0.1, 0.3, 0.5]:
    #             analyze_particle(p, c, n)
    analyze_particle(p=5, c=0.6, n=0.3)
