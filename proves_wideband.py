
# https://pyhank.readthedocs.io/en/latest/auto_examples/simple_example.html
# https://www.mdpi.com/2304-6732/12/4/306

#pip install pyhank
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from pyhank import HankelTransform
import torch

import torchoptics
from torchoptics.profiles import gaussian, square, circle
from torchoptics.elements import PhaseModulator
from torchoptics import Field, System, visualize_tensor
device = "cuda" if torch.cuda.is_available() else "cpu"


def KtoR(k, waveLength, focalLength):
    return waveLength * focalLength * k / (2 * np.pi)

def RtoK(r, waveLength, focalLength):
    return 2 * np.pi * r / (waveLength * focalLength)

def MSE_w(goalProfile, iterProfile, weights):

    # Donem mes pes als valors mes allunyats del centre, ja que son els que influiran
    # mes en fer la revolucio del perfil

    return np.sum(weights * (goalProfile - iterProfile)**2)/(np.sum(weights * goalProfile**2) + 1e-15)

def GerchbergSaxton(inputProfile, goalFunction, transformer, initial_doe, radius = 100, maxiter = 10000, errMin = 1e-4, initialSlope = 8):

    # define transformer & scale
    N = len(goalFunction)
    r = transformer.r
    kr = transformer.kr

    # normalize target intensity
    P_in = 2 * np.pi * np.trapz(np.abs(inputProfile)**2 * r, r)
    P_goal = 2 * np.pi * np.trapz(np.abs(goalFunction)**2 * r, r)
    goalProfile = goalFunction * np.sqrt(P_in/P_goal)


    n = 2
    if len(initial_doe)==0 :
        testfit = np.array([initialSlope*item*1e2 * (item*1e2)**n / ((item * 1e2)**n + 1) - 2.5 for item in r])
        # define initial DOE
        U_DOE = np.exp(1j * testfit)
    else:
        U_DOE = initial_doe

    # definim ponderacions per MSE
    #slope = 1
    #weights = np.linspace(1, N*slope, N)
    weights = kr * np.gradient(kr) # gradient per no uniforme

    # CHAT---------------------------------------

    # Output of initial field
    U_initial = transformer.qdht(inputProfile)

    # Integration weights in Hankel frequency coordinates
    weights = kr * np.gradient(kr)

    # Match target power to the initial transformed field
    P_in = np.sum(weights * np.abs(U_initial)**2)
    P_goal = np.sum(weights * np.abs(goalFunction)**2)

    goalProfile = goalFunction * np.sqrt(P_in / P_goal)

    # ------------------

    for i in range(maxiter):

        # --- DOE PLANE

        # apply DOE phase
        U_tot = inputProfile * U_DOE

        # propagate to focal plane
        U_image = transformer.qdht(U_tot)

        # ---- IMAGE PLANE

        # Find new intensity
        U_im_abs = np.abs(U_image)
        I_found = U_im_abs**2

        # check for MSE with radial weighting
        error = MSE_w(goalProfile, U_im_abs, weights)

        # replace amplitude, retain phase
        U_image = goalProfile * np.exp(1j * np.angle(U_image))

        # Check for flatness
        #flatness = (max(I) - min(I))/(max(I) + min(I))

        if error < errMin:
            print("Converged with error < errMin")
            break

        # go back to DOE
        U_DOE = transformer.iqdht(U_image)
        U_DOE = np.exp(1j * np.angle(U_DOE))

    print("# iterations = ", i)
    print("Error = ", error)
    return U_DOE


def wrap(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


class GS_DOE:
    def __init__(self, params):
        self.setParams(params)

    def setParams(self, params):
        self.fl = params["fl"]
        self.wl = params["wl"]
        self.rmax = params["rmax"]
        self.N = params["N"]
        self.w0 = params["w0"]
        self.wgoal = params["wgoal"]

    def setFocal(self, focal):
        self.fl = focal

    def setWl(self, wl):
        self.wl = wl

    def setW0(self, w0):
        self.w0 = w0

    def setrmax(self, rmax):
        self.rmax = rmax

    def makeProfiles(self, p=3):
        self.transformer = HankelTransform(order=0, max_radius=self.rmax, n_points=self.N)
        self.r = self.transformer.r
        self.kr = self.transformer.kr

        # define input intensity profile
        self.inputAmp = np.exp(-(self.r / self.w0) ** 2)
        # self.inputAmp = np.sqrt(self.inputInt)

        # UNUSED, forma general per separacio irregular
        # U_in = interp1d(
        #    self.r, self.inputAmp,
        #    bounds_error=False,
        #    fill_value=0
        # )(r)

        # Convert spatial frequency to physical focal-plane radius
        self.rho = self.wl * self.fl * self.transformer.kr / (2 * np.pi)

        self.goalProfile = np.exp(-(self.rho / self.wgoal) ** (2 * p))

    def showInputs(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(self.r * 1e2, self.inputAmp, label="Input amplitude")
        ax.plot(self.rho * 1e2, self.goalProfile, label="Goal amplitude")
        ax.legend(fontsize=14)
        ax.set_xlabel("Radius (cm)", fontsize=14)
        ax.set_ylabel("I/I$_0$", fontsize=14)
        ax.set_xlim(0, np.max(self.r) * 1e2)
        plt.show()

    def makeDOE(self,initial_doe, initialSlope=8, maxiter=10000, errMin=1e-4):
        self.initialSlope = initialSlope
        self.DOE = GerchbergSaxton(self.inputAmp, self.goalProfile, self.transformer, initial_doe, radius=self.rmax, maxiter=maxiter,
                                   errMin=errMin, initialSlope=initialSlope)

    def showResults(self, wrapped=False):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        testfit = np.array(
            [self.initialSlope * item * 1e2 * (item * 1e2) ** 2 / ((item * 1e2) ** 2 + 1) - 2.5 for item in self.r])

        if wrapped:
            axs[0].plot(self.r * 1e2, np.angle(self.DOE), label="DOE")
            testfit = wrap(testfit)
            axs[0].plot(self.r * 1e2, testfit, label="Initial fit")
        else:
            axs[0].plot(self.r * 1e2, np.unwrap(np.angle(self.DOE)), label="DOE")
            axs[0].plot(self.r * 1e2, testfit, label="Initial fit")
        axs[0].set_title("DOE phase profile")
        axs[0].set_xlabel("r (cm)", fontsize=14)
        axs[0].set_ylabel("Phase (rad)", fontsize=14)
        axs[0].set_xlim(0, np.max(self.r) * 1e2)
        axs[0].legend()

        # test DOE
        U_converged = self.inputAmp * self.DOE
        U_image = self.transformer.qdht(U_converged)

        normImage = np.abs(U_image) / max(np.abs(U_image))
        goalImage = np.abs(self.goalProfile) / max(np.abs(self.goalProfile))

        axs[1].plot(self.rho * 1e2, normImage ** 2, label="Converged")
        axs[1].plot(self.rho * 1e2, self.goalProfile ** 2, label="Goal")
        axs[1].legend(fontsize=14)
        axs[1].set_title("Image profile")
        axs[1].set_xlabel("Distance to center (cm)", fontsize=14)
        axs[1].set_ylabel("I/I$_0$", fontsize=14)
        axs[1].set_xlim(0, self.rmax * 1e2)
        axs[1].set_ylim(-0.1, 1.1)
        axs[1].vlines(self.wgoal * 1e2, -0.2, 1.2, ls="--", color="black")
        fig.show()

    def getDOE(self):
        return self.DOE



def thickness(phase, n, wl):
    return wl * phase / (2 * np.pi * (n-1))

def radial_to_2d(profile):
    n = len(profile)
    y, x = np.ogrid[-(n-1):n, -(n-1):n]
    r = np.sqrt(x**2 + y**2)
    return profile[np.clip(np.rint(r).astype(int), 0, n-1)]

def genLensPhase(N, wl, fl, dx):
    y, x = np.ogrid[-(N-1):N, -(N-1):N]
    r = np.sqrt(x**2 + y**2) * dx
    return -np.pi * r**2 / (wl * fl)

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


f = 300*1e9

does = []

for i in range(0,6, 1):
    f = f + i*1e9

    if i == 0:
        initial_doe = []
    else:
        initial_doe = does[-1]

    params = {
        "fl": 15e-2,  # cm
        "wl": 3e8 / f,
        "rmax": 7e-2,  # cm
        "N": 350,
        "w0": 0.02,  # cm
        "wgoal": 0.008  # cm
    }

    DOE_maker = GS_DOE(params)
    DOE_maker.makeProfiles(p=4)
    DOE_maker.showInputs()
    DOE_maker.makeDOE(initial_doe, initialSlope=4, maxiter=10000)

    does.append(np.unwrap(np.angle(DOE_maker.getDOE())))

radi1, prova1 = create_interspersed_doe(does)
prova1 = prova1[:-1, :-1]
radi2, prova2 = create_interspersed_doe(does, True)
prova2 = prova2[:-1, :-1]
prova3 = create_sector_doe(does, 360/60)
prova3 = prova3[:-1, :-1]

fig, ax = plt.subplots(1,3,figsize = (10,3))
ax[0].imshow(prova1)
ax[0].set_title("DOE 1")

ax[1].imshow(prova2)
ax[1].set_title("DOE 2")

ax[2].imshow(prova3)
ax[2].set_title("DOE 3")

plt.show()


for i in range(0,6, 1):
    f = f + i * 1e9

    if i == 0:
        initial_doe = []
    else:
        initial_doe = does[-1]

    print(initial_doe)
    params = {
        "fl": 15e-2,  # cm
        "wl": 3e8 / f,
        "rmax": 7e-2,  # cm
        "N": 350,
        "w0": 0.02,  # cm
        "wgoal": 0.008  # cm
    }

    dx = params['rmax'] / params['N']
    torchoptics.set_default_spacing(dx)
    torchoptics.set_default_wavelength(params['wl'])
    shape = 2 * params['N'] - 1


    # Prova 1
    ph_lens = genLensPhase(params['N'], params['wl'], params['fl'], dx)
    DOE_2D = torch.from_numpy(prova1) + ph_lens

    waist_radius = params['w0']
    source_profile = gaussian(shape, waist_radius=waist_radius).to(device)

    input_field = Field(source_profile, z=0).to(device)
    system = System(
        # Detach from any computation history, load directly as final phase
        PhaseModulator(DOE_2D, z=0),
    ).to(device)

    output_field_1 = system.measure_at_z(input_field, params['fl'])

    # Prova 2
    ph_lens = genLensPhase(params['N'], params['wl'], params['fl'], dx)
    DOE_2D = torch.from_numpy(prova2) + ph_lens

    waist_radius = params['w0']
    source_profile = gaussian(shape, waist_radius=waist_radius).to(device)

    input_field = Field(source_profile, z=0).to(device)
    system = System(
        # Detach from any computation history, load directly as final phase
        PhaseModulator(DOE_2D, z=0),
    ).to(device)

    output_field_2 = system.measure_at_z(input_field, params['fl'])

    """ """
    # Prova 3
    ph_lens = genLensPhase(params['N'], params['wl'], params['fl'], dx)
    DOE_2D = torch.from_numpy(prova3) + ph_lens

    waist_radius = params['w0']
    source_profile = gaussian(shape, waist_radius=waist_radius).to(device)

    input_field = Field(source_profile, z=0).to(device)
    system = System(
        # Detach from any computation history, load directly as final phase
        PhaseModulator(DOE_2D, z=0),
    ).to(device)

    output_field_3 = system.measure_at_z(input_field, params['fl'])

    # Prova 4
    ph_lens = genLensPhase(params['N'], params['wl'], params['fl'], dx)
    DOE_2D = torch.from_numpy(radial_to_2d(does[0])) + ph_lens

    waist_radius = params['w0']
    source_profile = gaussian(shape, waist_radius=waist_radius).to(device)

    input_field = Field(source_profile, z=0).to(device)
    system = System(
        # Detach from any computation history, load directly as final phase
        PhaseModulator(DOE_2D, z=0),
    ).to(device)

    output_field_4 = system.measure_at_z(input_field, params['fl'])

    # Prova 5
    ph_lens = genLensPhase(params['N'], params['wl'], params['fl'], dx)
    DOE_2D = torch.from_numpy(radial_to_2d(does[-1])) + ph_lens

    waist_radius = params['w0']
    source_profile = gaussian(shape, waist_radius=waist_radius).to(device)

    input_field = Field(source_profile, z=0).to(device)
    system = System(
        # Detach from any computation history, load directly as final phase
        PhaseModulator(DOE_2D, z=0),
    ).to(device)

    output_field_5 = system.measure_at_z(input_field, params['fl'])

    # Prova 6
    ph_lens = genLensPhase(params['N'], params['wl'], params['fl'], dx)
    DOE_2D = torch.from_numpy(radial_to_2d(does[i])) + ph_lens

    waist_radius = params['w0']
    source_profile = gaussian(shape, waist_radius=waist_radius).to(device)

    input_field = Field(source_profile, z=0).to(device)
    system = System(
        # Detach from any computation history, load directly as final phase
        PhaseModulator(DOE_2D, z=0),
    ).to(device)

    output_field_6 = system.measure_at_z(input_field, params['fl'])

    fig, ax = plt.subplots(2,3,figsize = (10,3))
    ax[0,0].imshow(output_field_1.intensity().detach().numpy())
    ax[0,0].set_title("Output intensity 1")

    ax[0,1].imshow(output_field_2.intensity().detach().numpy())
    ax[0,1].set_title("Output intensity 2")

    ax[0,2].imshow(output_field_3.intensity().detach().numpy())
    ax[0,2].set_title("Output intensity 3")

    ax[1,0].imshow(output_field_4.intensity().detach().numpy())
    ax[1,0].set_title("Output intensity 4")

    ax[1,1].imshow(output_field_5.intensity().detach().numpy())
    ax[1,1].set_title("Output intensity 5")

    ax[1,2].imshow(output_field_6.intensity().detach().numpy())
    ax[1,2].set_title("Output intensity 6")

    plt.show()



print('Fi')
