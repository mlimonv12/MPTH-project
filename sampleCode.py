import torch
import torchoptics
from torch.nn import Parameter
from torchoptics import Field, System
from torchoptics.elements import PhaseModulator
from torchoptics.profiles import gaussian, laguerre_gaussian

device = "cuda" if torch.cuda.is_available() else "cpu"
torchoptics.set_default_spacing(10e-6)
torchoptics.set_default_wavelength(700e-9)

shape = 250
waist_radius = 300e-6

input_field = Field(gaussian(shape, waist_radius=waist_radius), z=0).to(device)

petal_profile = laguerre_gaussian(shape, p=0, l=4, waist_radius=waist_radius)
petal_profile += laguerre_gaussian(shape, p=0, l=-4, waist_radius=waist_radius)
target_field = Field(petal_profile, z=0.8).normalize().to(device)

system = System(
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.2),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.4),
    PhaseModulator(Parameter(torch.zeros(shape, shape)), z=0.6),
).to(device)

optimizer = torch.optim.Adam(system.parameters(), lr=0.05)
for iteration in range(100):
    optimizer.zero_grad()
    output_field = system.measure_at_z(input_field, 0.8)
    loss = 1 - output_field.inner(target_field).abs().square()
    loss.backward()
    optimizer.step()