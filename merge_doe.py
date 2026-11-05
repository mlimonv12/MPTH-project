import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyhank import HankelTransform
import torch
import torchoptics
from torchoptics.profiles import gaussian, square, circle
from torchoptics.elements import PhaseModulator
from torchoptics import Field, System, visualize_tensor

import numpy as np


def create_interspersed_doe(DOEs: list, mirror: bool = False):
    """
    Merges radial profile information from multiple DOE lenses into a
    radially symmetric DOE phase/height map (2D circular lens).

    Parameters
    ----------
    DOEs : list or np.ndarray
        List of 1D arrays, where DOEs[i] contains the radial profile
        information for lens i.
    mirror : bool, default=False
        If False, intersperses in standard periodic sequence: a, b, c, a, b, c...
        If True, intersperses in mirrored sequence: a, b, c, c, b, a, a, b, c, c, b, a...

    Returns
    -------
    merged_radius : np.ndarray
        1D array of length `radius_len` containing the interspersed radial profile.
    doe_grid : np.ndarray
        2D circular image of shape (2*radius_len, 2*radius_len) formed by
        revolving the 1D radial profile around the center.
    """
    DOEs = [np.asarray(doe) for doe in DOEs]
    n = len(DOEs)

    if n == 0:
        raise ValueError("DOEs array must contain at least one lens profile.")

    # 1. Determine radial length and grid size
    radius_len = len(DOEs[0])
    g = 2 * radius_len  # Grid dimension based on radius length

    # 2. Interspersed radial sequence generation (1D profile)
    merged_radius = np.zeros(radius_len)

    for r in range(radius_len):
        if mirror:
            cycle_len = 2 * n
            rem = r % cycle_len
            lens_idx = rem if rem < n else (cycle_len - 1 - rem)
        else:
            lens_idx = r % n

        profile = DOEs[lens_idx]
        merged_radius[r] = profile[r] if r < len(profile) else profile[-1]

    # 3. Perform 360-degree radial revolution to create the 2D circular lens
    center = (g - 1) / 2.0
    y, x = np.ogrid[:g, :g]
    r_grid = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Map radial distance to the nearest index in merged_radius
    r_indices = np.round(r_grid).astype(int)
    r_indices = np.clip(r_indices, 0, radius_len - 1)

    doe_grid = merged_radius[r_indices]

    return merged_radius, doe_grid

def create_sector_doe(DOEs: list, a: float):
    """
    Merges radial profile information from multiple DOE lenses into a
    radially symmetric DOE phase map divided into fine-grained angular sectors.

    Parameters
    ----------
    DOEs : list or np.ndarray
        List of 1D arrays, where DOEs[i] contains the radial profile
        information for lens i.
    a : float
        The angle (in degrees) of each circular sector. Can be fractional.
        360 / a must result in an integer that is a multiple of len(DOEs).

    Returns
    -------
    doe_grid : np.ndarray
        2D circular image of shape (2*radius_len, 2*radius_len) formed by
        assigning angular sectors to consecutive DOE profiles.
    """
    DOEs = [np.asarray(doe) for doe in DOEs]
    n = len(DOEs)

    if n == 0:
        raise ValueError("DOEs array must contain at least one lens profile.")

    # 1. Floating-point safe validation for the fractional angle parameter 'a'
    num_sectors_exact = 360.0 / a
    num_sectors = int(np.round(num_sectors_exact))

    # Use np.isclose to avoid rejecting valid floats due to precision errors
    # (e.g., 360 / 0.1 becoming 3599.9999999999995)
    if not np.isclose(num_sectors_exact, num_sectors, atol=1e-9):
        raise ValueError(
            f"360 must be evenly divisible by angle 'a'. "
            f"Got 360 / {a} = {num_sectors_exact}"
        )

    if num_sectors % n != 0:
        raise ValueError(
            f"The number of sectors ({num_sectors}) must be a multiple "
            f"of the number of DOEs ({n})."
        )

    # 2. Determine radial length and grid size
    radius_len = len(DOEs[0])
    g = 2 * radius_len

    # 3. Create 2D coordinate grid (Cartesian to Polar)
    center = (g - 1) / 2.0
    y, x = np.ogrid[:g, :g]

    # Calculate radial distance for each pixel
    r_grid = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_indices = np.round(r_grid).astype(int)
    r_indices = np.clip(r_indices, 0, radius_len - 1)

    # Calculate angular position for each pixel (in degrees, 0 to 360)
    theta_grid = np.arctan2(y - center, x - center)
    theta_deg = np.degrees(theta_grid) % 360

    # 4. Determine which lens each pixel belongs to using float division and floor
    sector_idx = np.floor(theta_deg / a).astype(int)

    # Safety catch to ensure sector_idx exactly matches num_sectors at 359.999 degrees
    sector_idx = np.clip(sector_idx, 0, num_sectors - 1)

    lens_idx = sector_idx % n

    # 5. Map values from the corresponding DOE using advanced 2D indexing
    doe_stack = np.vstack(DOEs)
    doe_grid = doe_stack[lens_idx, r_indices]

    return doe_grid


import numpy as np
import itertools


def create_permuted_sector_doe(DOEs: list, a: float):
    """
    1. Creates n! auxiliary concentric radii from all permutations of the input DOEs.
    2. Maps these auxiliary radii onto angular sectors to form a 2D lens.

    Parameters
    ----------
    DOEs : list or np.ndarray
        List of 1D arrays, where DOEs[i] contains the radial profile.
    a : float
        The angle (in degrees) of each circular sector. Can be fractional.
        360 / a must result in an integer that is a multiple of len(DOEs)!.

    Returns
    -------
    doe_grid : np.ndarray
        2D circular image of shape (2*radius_len, 2*radius_len).
    """
    DOEs = [np.asarray(doe) for doe in DOEs]
    n = len(DOEs)

    if n == 0:
        raise ValueError("DOEs array must contain at least one lens profile.")

    # Standardize lengths to ensure vectorization works safely
    radius_len = len(DOEs[0])
    padded_DOEs = []
    for doe in DOEs:
        if len(doe) < radius_len:
            # Pad with the last value if too short
            padded_DOEs.append(np.pad(doe, (0, radius_len - len(doe)), mode='edge'))
        else:
            # Clip if too long
            padded_DOEs.append(doe[:radius_len])

    orig_stack = np.vstack(padded_DOEs)

    # 1. Generate all permutations of lens indices (0 to n-1)
    # For n=3, this creates (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
    perms = list(itertools.permutations(range(n)))
    num_perms = len(perms)

    # 2. Vectorized generation of auxiliary radii
    r_idx = np.arange(radius_len)
    base_lens_indices = r_idx % n  # Sequence: 0, 1, 2, 0, 1, 2...

    aux_DOEs = []
    for p in perms:
        p_array = np.array(p)
        # Translate the base sequence into the permuted sequence
        permuted_lens_indices = p_array[base_lens_indices]
        # Fetch the radial values for this specific permutation
        aux_radius = orig_stack[permuted_lens_indices, r_idx]
        aux_DOEs.append(aux_radius)

    # 3. Validate angle parameter 'a' against the number of PERMUTATIONS (n!)
    num_sectors_exact = 360.0 / a
    num_sectors = int(np.round(num_sectors_exact))

    if not np.isclose(num_sectors_exact, num_sectors, atol=1e-9):
        raise ValueError(
            f"360 must be evenly divisible by angle 'a'. "
            f"Got 360 / {a} = {num_sectors_exact}"
        )

    if num_sectors % num_perms != 0:
        raise ValueError(
            f"The number of sectors ({num_sectors}) must be a multiple "
            f"of the number of auxiliary permutations ({num_perms} for n={n})."
        )

    # 4. Create 2D coordinate grid (Cartesian to Polar)
    g = 2 * radius_len
    center = (g - 1) / 2.0
    y, x = np.ogrid[:g, :g]

    # Radial distances
    r_grid = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_indices = np.round(r_grid).astype(int)
    r_indices = np.clip(r_indices, 0, radius_len - 1)

    # Angular positions
    theta_grid = np.arctan2(y - center, x - center)
    theta_deg = np.degrees(theta_grid) % 360

    # 5. Determine which auxiliary radius each pixel belongs to
    sector_idx = np.floor(theta_deg / a).astype(int)
    sector_idx = np.clip(sector_idx, 0, num_sectors - 1)

    # Map the sector to one of the generated permutations
    aux_idx = sector_idx % num_perms

    # 6. Map values from the auxiliary DOEs onto the 2D grid
    aux_stack = np.vstack(aux_DOEs)
    doe_grid = aux_stack[aux_idx, r_indices]

    return doe_grid

doe1 = arr = np.full(200, 10)
doe2 = arr = np.full(200, 0)
doe3 = arr = np.full(200, -10)
does = [doe1, doe2, doe3]
radi1, prova1 = create_interspersed_doe(does)
radi2, prova2 = create_interspersed_doe(does, True)
does2 = [radi1, radi2, doe3]
prova3 = create_sector_doe(does, 360/300)
prova4 = create_sector_doe(does2, 360/300)
prova5 = create_permuted_sector_doe(does, 360/120)
plt.imshow(prova1)
#print(radi1)
#print('\n')
#print(radi2)
plt.show()
plt.imshow(prova2)
plt.show()
plt.imshow(prova3)
plt.show()
plt.imshow(prova4)
plt.show()
plt.imshow(prova5)
plt.show()
print('Fi')