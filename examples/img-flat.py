#!/usr/bin/env python3

import mim
import numpy as np
import matplotlib.pyplot as plt

prng = mim.Prng();

f = lambda x: 100 - 3.5*x**2

par = 1.8
shape = (200, 100)

def randomize(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j] = prng.poisson(data[i,j])

# Observation
obs = np.full(shape, f(1.8))
randomize(obs)

# Parameter
parameter = np.linspace(1.0, 3.0, 50)

# Model
images = np.stack(
    [np.full(shape, f(x)) for x in parameter]
)

"""
for i, image in enumerate(images):
    randomize(image)
"""

model = mim.Model(parameter, images)

print(f"shape = {model.shape}")
print(f"pmin  = {model.pmin}")
print(f"pmax  = {model.pmax}")

min_value = 800
image, bin_image, value_image = model.invert_min(obs, min_value)

plt.figure(figsize=(12, 7))
plt.imshow(image.T, cmap=plt.cm.jet)
plt.colorbar()
plt.show()
