import torch
import torch.fft as fft
import torchoptics
from torch.nn import Parameter
from torchoptics import Field, System, visualize_tensor
from torchoptics.elements import PhaseModulator
from torchoptics.profiles import gaussian, square
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Physics & Grid Setup (Scaled for mm-waves)
# ==========================================
c = 299792458.0  # Speed of light in m/s

# Set frequencies: 6 steps from 200 GHz to 460 GHz
frequencies = np.linspace(200e9, 460e9, 6).tolist()
f0 = 330e9  # Center Reference Frequency (330 GHz)
lam_0 = c / f0  # ~0.909 mm

# Scaled grid parameters for ~1mm wavelengths
dx = 0.2e-3  # 0.2 mm pixel size (must be < lambda/2)
shape = 600  # 600x600 grid (120x120 mm total area)
waist_radius = 30e-3  # 30 mm incident beam radius
target_pos = 150e-3  # 150 mm propagation distance

torchoptics.set_default_spacing(dx)
torchoptics.set_default_wavelength(lam_0)

pixels = 60
square_side = pixels * dx  # 12x12 mm target square

# Generate raw profile tensors
source_profile = gaussian(shape, waist_radius=waist_radius).to(device)
target_profile = square(shape, side=square_side).to(device)

# Normalize target amplitude
total_source_irradiance = torch.sum(torch.abs(source_profile) ** 2)
total_target_irradiance = torch.sum(torch.abs(target_profile) ** 2)
scaling_factor = torch.sqrt(total_source_irradiance / total_target_irradiance)
A_tgt = target_profile * scaling_factor

# Initialize Random Binary Phase (0 or pi) at the reference frequency
torch.manual_seed(0)
phi = torch.randint(0, 2, (shape, shape), device=device, dtype=torch.float32) * torch.pi

# --- Pre-calculate Angular Spectrum Transfer Functions ---
lambdas = [c / f for f in frequencies]

fx = fft.fftfreq(shape, d=dx, device=device)
fy = fft.fftfreq(shape, d=dx, device=device)
FX, FY = torch.meshgrid(fx, fy, indexing='ij')

H_fws = []
H_bws = []

for lam in lambdas:
    term = 1 - (lam * FX) ** 2 - (lam * FY) ** 2
    mask = term >= 0  # Filter out evanescent waves
    phase_term = 2 * torch.pi * target_pos / lam * torch.sqrt(term * mask)

    H_fws.append(torch.exp(1j * phase_term) * mask)
    H_bws.append(torch.exp(-1j * phase_term) * mask)

efficiencies = []

# ==========================================
# 2. Sequential Broadband Optimization Loop
# ==========================================
for iteration in range(100):
    eff_total = 0.0

    for i, (f, lam, H_fw, H_bw) in enumerate(zip(frequencies, lambdas, H_fws, H_bws)):
        # Scale phase for current frequency dispersion
        phi_f = phi * (f / f0)

        # 1. Modulate incident wave
        E1 = source_profile * torch.exp(1j * phi_f)

        # 2. Forward propagate
        E2 = fft.ifft2(fft.fft2(E1) * H_fw)

        # Calculate Efficiency
        target_mask = (target_profile > 0).float()
        power_in_target = torch.sum((torch.abs(E2) ** 2) * target_mask)
        eff = (power_in_target / total_source_irradiance).item()
        eff_total += eff

        # 3. Modify Target
        E20 = A_tgt * torch.exp(1j * torch.angle(E2))

        # 4. Backward propagate
        E10 = fft.ifft2(fft.fft2(E20) * H_bw)

        # 5. Calculate Sensitivity Metric for THIS frequency
        ds = torch.real(E1 * E10)

        # 6. IMMEDIATE Phase Update Rule (SGD style)
        toggle_mask = ds < 0
        phi = torch.where(toggle_mask, torch.pi - phi, phi)

    avg_eff = eff_total / len(frequencies)
    efficiencies.append(avg_eff)

    if iteration > 0 and abs(avg_eff - efficiencies[-2]) <= 1e-4:
        print(f"Converged at iteration {iteration}. Avg Efficiency: {avg_eff:.4f}")
        break

    if iteration % 10 == 0:
        print(f"Iteration {iteration} | Avg Efficiency: {avg_eff:.4f}")

# ==========================================
# 3. Data Collection & Plotting
# ==========================================
intensities_per_freq = []

with torch.no_grad():
    for f, H_fw in zip(frequencies, H_fws):
        phi_f = phi * (f / f0)
        E1 = (source_profile * torch.exp(1j * phi_f)).to(torch.complex64)
        E2 = fft.ifft2(fft.fft2(E1) * H_fw)

        I_f = (torch.abs(E2) ** 2).to(torch.float32).cpu().numpy()
        intensities_per_freq.append(I_f)

phi_np = phi.cpu().numpy()
goal_profile_np = target_profile.cpu().numpy()

print("\nOptimization Complete. Generating plots...")

# --- Plot the final freeform binary 2D Phase Profile ---
fig1, ax1 = plt.subplots(1, 1, figsize=(5.5, 4))
im = ax1.imshow(phi_np, cmap='binary_r', vmin=0, vmax=np.pi)
ax1.set_title("Optimized 2D Phase Profile (200-460 GHz)")
cbar = plt.colorbar(im, ax=ax1, ticks=[0, np.pi])
cbar.ax.set_yticklabels(['0', r'$\pi$'])
plt.tight_layout()
plt.show()

# --- Plot Cross-Section Irradiances for EACH Frequency ---
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
axes2 = axes2.flatten()

center_idx = shape // 2
distance_mm = (np.arange(shape) - center_idx) * dx * 1000
goal_norm = goal_profile_np / np.max(goal_profile_np)

for i, (I_f, f) in enumerate(zip(intensities_per_freq, frequencies)):
    ax = axes2[i]

    I_norm = I_f / np.max(I_f)

    ax.plot(distance_mm, I_norm[center_idx, :], color='forestgreen', linestyle='-', label='x direction')
    ax.plot(distance_mm, I_norm[:, center_idx], color='forestgreen', linestyle=':', label='y direction')
    ax.plot(distance_mm, goal_norm[center_idx, :], color='red', linestyle='--', linewidth=1.5, label='Goal profile')

    # Plot titles scaled to GHz
    f_ghz = f / 1e9

    ax.set_title(f"Irradiance @ {f_ghz:.1f} GHz")
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Irradiance (a.u.)")

    # Zoom in to the central 40mm to see the 12mm square clearly
    ax.set_xlim([-20.0, 20.0])

    if i == 0:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.show()