import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import itertools

import torch
import torch.fft as fft
import torchoptics
from torchoptics.profiles import gaussian, circle
from torchoptics.elements import PhaseModulator
from torchoptics import Field, System
# Ensure pyhank is installed (pip install pyhank)
from pyhank import HankelTransform

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# PROGRAM 1: FUNCTIONS & 1D BINARY INITIALIZATION
# ==========================================

def Binary_IFTA_1D(inputProfile, goalFunction, transformer, maxiter=150):
    """
    1D Phase optimization constrained to {0, pi} using the metric provided.
    """
    kr = transformer.kr
    weights = kr * np.gradient(kr)

    # Power Normalization
    U_initial = transformer.qdht(inputProfile)
    P_in = np.sum(weights * np.abs(U_initial) ** 2)
    P_goal = np.sum(weights * np.abs(goalFunction) ** 2)
    A_tgt = goalFunction * np.sqrt(P_in / P_goal)

    # Initialize with random binary phase {0, pi}
    phi = np.random.choice([0.0, np.pi], size=inputProfile.shape)

    for i in range(maxiter):
        # 1. Modulate field
        E1 = inputProfile * np.exp(1j * phi)

        # 2. Forward propagate
        E2 = transformer.qdht(E1)

        # 3. Modify target: Replace amplitude, maintain phase
        E20 = A_tgt * np.exp(1j * np.angle(E2))

        # 4. Backward propagate
        E10 = transformer.iqdht(E20)

        # 5. Calculate Sensitivity Metric
        ds = np.real(E1 * E10)

        # 6. Phase Update Rule (Toggle where ds < 0)
        toggle_mask = ds < 0
        phi = np.where(toggle_mask, np.pi - phi, phi)

    return phi


class Binary_DOE_1D:
    def __init__(self, params):
        self.fl = params["fl"]
        self.wl = params["wl"]
        self.rmax = params["rmax"]
        self.N = params["N"]
        self.w0 = params["w0"]
        self.wgoal = params["wgoal"]

    def makeProfiles(self, p=4):
        self.transformer = HankelTransform(order=0, max_radius=self.rmax, n_points=self.N)
        self.r = self.transformer.r
        self.kr = self.transformer.kr

        self.inputAmp = np.exp(-(self.r / self.w0) ** 2)
        self.rho = self.wl * self.fl * self.transformer.kr / (2 * np.pi)
        self.goalProfile = np.exp(-(self.rho / self.wgoal) ** (2 * p))

    def makeDOE(self, maxiter=150):
        self.DOE = Binary_IFTA_1D(self.inputAmp, self.goalProfile, self.transformer, maxiter=maxiter)

    def getDOE(self):
        return self.DOE


def radial_to_2d(profile):
    n = len(profile)
    y, x = np.ogrid[-(n - 1):n, -(n - 1):n]
    r = np.sqrt(x ** 2 + y ** 2)
    return profile[np.clip(np.rint(r).astype(int), 0, n - 1)]


def genLensPhase(N, wl, fl, dx):
    y, x = np.ogrid[-(N - 1):N, -(N - 1):N]
    r = np.sqrt(x ** 2 + y ** 2) * dx
    return -np.pi * r ** 2 / (wl * fl)


def create_interspersed_doe(DOEs: list, mirror: bool = False):
    DOEs = [np.asarray(doe) for doe in DOEs]
    n = len(DOEs)
    radius_len = len(DOEs[0])
    g = 2 * radius_len
    merged_radius = np.zeros(radius_len)

    for r in range(radius_len):
        lens_idx = r % n
        profile = DOEs[lens_idx]
        merged_radius[r] = profile[r] if r < len(profile) else profile[-1]

    center = (g - 1) / 2.0
    y, x = np.ogrid[:g, :g]
    r_grid = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_indices = np.clip(np.round(r_grid).astype(int), 0, radius_len - 1)
    return merged_radius, merged_radius[r_indices]


def create_permuted_sector_doe(DOEs: list, a: float):
    DOEs = [np.asarray(doe) for doe in DOEs]
    n = len(DOEs)
    radius_len = len(DOEs[0])
    padded_DOEs = [np.pad(doe, (0, radius_len - len(doe)), mode='edge') if len(doe) < radius_len else doe[:radius_len]
                   for doe in DOEs]

    orig_stack = np.vstack(padded_DOEs)
    perms = list(itertools.permutations(range(n)))
    num_perms = len(perms)
    r_idx = np.arange(radius_len)

    aux_DOEs = [orig_stack[np.array(p)[r_idx % n], r_idx] for p in perms]

    num_sectors = int(np.round(360.0 / a))
    g = 2 * radius_len
    center = (g - 1) / 2.0
    y, x = np.ogrid[:g, :g]

    r_grid = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_indices = np.clip(np.round(r_grid).astype(int), 0, radius_len - 1)

    theta_grid = np.arctan2(y - center, x - center)
    theta_deg = np.degrees(theta_grid) % 360

    sector_idx = np.clip(np.floor(theta_deg / a).astype(int), 0, num_sectors - 1)
    aux_idx = sector_idx % num_perms
    aux_stack = np.vstack(aux_DOEs)

    return aux_stack[aux_idx, r_indices]


# ---------------------------------------------
# Generate Initial 1D Profiles
# ---------------------------------------------
print("Generating 1D DOEs with Binary Phases...")
f0 = 200 * 1e9
does = []

np.random.seed(42)

for i in range(6):
    f = f0 + i * 30 * 1e9
    params = {
        "fl": 15e-2, "wl": 3e8 / f, "rmax": 10e-2,
        "N": 350, "w0": 0.02, "wgoal": 0.008
    }
    DOE_maker = Binary_DOE_1D(params)
    DOE_maker.makeProfiles(p=4)
    DOE_maker.makeDOE(maxiter=150)
    does.append(DOE_maker.getDOE())

# Create the 2D starting combinations
_, prova1 = create_interspersed_doe(does)
prova1 = prova1[:-1, :-1]

prova3 = create_permuted_sector_doe(does, 360 / 1440)
prova3 = prova3[:-1, :-1]

does_0_2d = radial_to_2d(does[0])
does_end_2d = radial_to_2d(does[-1])

shape_2d = 2 * params['N'] - 1
random_doe_2d = np.random.choice([0.0, np.pi], size=(shape_2d, shape_2d))

initial_DOEs = [prova1, prova3, does_0_2d, does_end_2d, random_doe_2d]
titles = ["Interspersed", "Permuted Sector", "1st Freq", "Last Freq", "Random Guess"]

# ==========================================
# PROGRAM 2: PYTORCH BINARY MULTI-FREQ OPTIMIZATION
# ==========================================
print("\nStarting 2D Multi-Frequency Binary Optimization...")
dx = params['rmax'] / params['N']
torchoptics.set_default_spacing(dx)

source_profile = gaussian(shape_2d, waist_radius=params['w0']).to(device, dtype=torch.float32)
target_profile = circle(shape_2d, radius=params['wgoal']).to(device, dtype=torch.float32)
target_mask = (target_profile > 0).to(torch.float32)

P_in = torch.sum(torch.abs(source_profile) ** 2)
P_tgt = torch.sum(target_mask)
scaling_factor = torch.sqrt(P_in / P_tgt)
A_tgt = (target_mask * scaling_factor).to(torch.float32)

frequencies = [f0 + i * 30 * 1e9 for i in range(6)]
H_fws, H_bws = [], []
fx = fft.fftfreq(shape_2d, d=dx, device=device).to(torch.float32)
fy = fft.fftfreq(shape_2d, d=dx, device=device).to(torch.float32)
FX, FY = torch.meshgrid(fx, fy, indexing='ij')

for f in frequencies:
    lam_f = 3e8 / f
    term = 1 - (lam_f * FX) ** 2 - (lam_f * FY) ** 2
    mask = term >= 0
    phase_term = 2 * torch.pi * params['fl'] / lam_f * torch.sqrt(torch.clamp(term, min=0))
    H_fws.append((torch.exp(1j * phase_term.to(torch.float32)) * mask).to(torch.complex64))
    H_bws.append((torch.exp(-1j * phase_term.to(torch.float32)) * mask).to(torch.complex64))

f_center = f0 + 2.5 * 30e9
base_lens = genLensPhase(params['N'], 3e8 / f_center, params['fl'], dx)
base_lens_t = torch.tensor(base_lens, device=device, dtype=torch.float32)

# --- NEW: Added list to store initial phases before optimization ---
initial_phases = []
optimized_phases, optimized_outputs = [], []

for init_doe, title in zip(initial_DOEs, titles):
    print(f"\nOptimizing Lens: {title}")

    # 1. Add geometrical base lens to the generated DOE.
    combined_phase = torch.tensor(init_doe, device=device, dtype=torch.float32) + base_lens_t

    # 2. Binarize the entire system strictly into {0, pi} for an accurate starting point.
    phi = torch.where(torch.cos(combined_phase) >= 0,
                      torch.zeros_like(combined_phase),
                      torch.full_like(combined_phase, torch.pi))

    # --- NEW: Capture the initial binary phase before the loop starts ---
    initial_phases.append(phi.cpu().numpy())

    num_iterations = 30

    for iteration in range(num_iterations):
        ds_total = torch.zeros_like(phi, dtype=torch.float32)
        mean_eff = 0

        for f_idx in range(len(frequencies)):
            E1 = (source_profile * torch.exp(1j * phi)).to(torch.complex64)
            E2 = fft.ifft2(fft.fft2(E1) * H_fws[f_idx])

            pow_in_tgt = torch.sum((torch.abs(E2) ** 2) * target_mask)
            mean_eff += (pow_in_tgt / P_in).item()

            E20 = A_tgt * torch.exp(1j * torch.angle(E2))
            E10 = fft.ifft2(fft.fft2(E20) * H_bws[f_idx])

            ds = torch.real(E1 * E10)
            ds_total += ds

        toggle_mask = ds_total < 0
        phi = torch.where(toggle_mask, torch.pi - phi, phi)

        if (iteration + 1) % 5 == 0:
            print(f"  Iter {iteration + 1}/{num_iterations} | Mean Efficiency: {mean_eff / len(frequencies):.4f}")

    optimized_phases.append(phi.cpu().numpy())

    avg_I = torch.zeros((shape_2d, shape_2d), device=device, dtype=torch.float32)
    with torch.no_grad():
        for f_idx in range(len(frequencies)):
            E1 = (source_profile * torch.exp(1j * phi)).to(torch.complex64)
            E2 = fft.ifft2(fft.fft2(E1) * H_fws[f_idx])
            avg_I += (torch.abs(E2) ** 2).to(torch.float32)
    avg_I /= len(frequencies)
    optimized_outputs.append(avg_I.cpu().numpy())

print("\nOptimization Complete. Generating plots...")

# ==========================================
# PLOTTING RESULTS
# ==========================================
num_lenses = len(titles)

# --- NEW: Plot the Initial Binary 2D Phase Profiles (Before Optimization) ---
fig0, axes0 = plt.subplots(1, num_lenses, figsize=(4.5 * num_lenses, 4))
for ax, phase, title in zip(axes0, initial_phases, titles):
    im = ax.imshow(phase, cmap='binary_r', vmin=0, vmax=np.pi)
    ax.set_title(f"Initial: {title}")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, np.pi])
    cbar.ax.set_yticklabels(['0', r'$\pi$'])
fig0.suptitle("Initial 2D Binary Phase Profiles {0, $\pi$} (Before Optimization)", fontsize=16)
plt.tight_layout()
plt.show()

# 1. Plot the final strictly binary 2D Phase Profiles (After Optimization)
fig1, axes1 = plt.subplots(1, num_lenses, figsize=(4.5 * num_lenses, 4))
for ax, phase, title in zip(axes1, optimized_phases, titles):
    im = ax.imshow(phase, cmap='binary_r', vmin=0, vmax=np.pi)
    ax.set_title(f"Optimized: {title}")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, np.pi])
    cbar.ax.set_yticklabels(['0', r'$\pi$'])
fig1.suptitle("Optimized 2D Binary Phase Profiles {0, $\pi$}", fontsize=16)
plt.tight_layout()
plt.show()

# 2. Plot Cross-Section Irradiances
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
axes2 = axes2.flatten()
for j in range(num_lenses, len(axes2)):
    axes2[j].axis('off')

center_idx = params['N'] - 1
distance_mm = np.arange(shape_2d) * dx * 1000
goal_profile = np.where(np.abs(distance_mm - distance_mm[center_idx]) <= params["wgoal"] * 1000, 1.0, 0.0)

for i, (avg_I, title) in enumerate(zip(optimized_outputs, titles)):
    I_norm = avg_I / np.max(avg_I)

    ax = axes2[i]
    ax.plot(distance_mm, I_norm[center_idx, :], color='forestgreen', linestyle='-', label='x direction')
    ax.plot(distance_mm, I_norm[:, center_idx], color='forestgreen', linestyle=':', label='y direction')
    ax.plot(distance_mm, goal_profile, color='red', linestyle='--', linewidth=1.5, label='Goal profile')

    ax.set_title(f"Mean Irradiance: {title}")
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Irradiance (a.u.)")
    if i == 0:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.show()