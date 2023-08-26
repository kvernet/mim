#!/usr/bin/env python3

import mim
import numpy as np
import matplotlib.pyplot as plt

prng = mim.Prng();

f = lambda x: 100 - 3.5*x**2

par = 1.8
shape = (200, 100)

def randomize(data):
    result = np.empty(shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[i,j] = prng.poisson(data[i,j])
    return result

# Observation
obs = np.full(shape, f(1.8))

# Parameter
parameter = np.linspace(1.0, 3.0, 50)

# Model
images = np.stack(
    [np.full(shape, f(x)) for x in parameter]
)

model = mim.Model(parameter, images)

print(f"shape = {model.shape}")
print(f"pmin  = {model.pmin}")
print(f"pmax  = {model.pmax}")

min_value, sigma, n = 800, 4, 10
img_sum = np.zeros(shape)
img_sum2 = np.zeros(shape)
for i in range(n):
    print(f"process {i+1}/{n} ...")
    robs = randomize(obs)
    img, _, _ = model.invert_min(robs, min_value, None, sigma)
    img_sum += img
    img_sum2 += img * img

img_mean = img_sum / n
img_rms = np.sqrt((img_sum2/n - img_mean*img_mean) / n)

# Save stats
mim.Data(img_mean).dump("img_mean.pkl.gz")
mim.Data(img_rms).dump("img_rms.pkl.gz")

# Plot stats
plt.figure(figsize=(12, 7))
plt.imshow(img_mean.T, cmap=plt.cm.jet)
plt.colorbar()

plt.figure(figsize=(12, 7))
plt.imshow(img_rms.T, cmap=plt.cm.jet)
plt.colorbar()

plt.show()
