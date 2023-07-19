#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plot
import mim

pmin, pmax = 1.0, 3.0
shape = (100, 200)

# Parametric model
f = lambda x: 9.1 - x**2

# Parameter
parameter = numpy.linspace(pmin, pmax, 11)

# Model
images = numpy.stack(
    [ numpy.full(shape, f(x)) for x in parameter ]
)
model = mim.Model(parameter, images)
print(f"shape = {model.shape}")
print(f"pmin  = {model.pmin}")
print(f"pmax  = {model.pmax}")


# Observation
obs = numpy.full(shape, f(1.81))

# Check interploation at nodes
for x in parameter:
    snapshot = model(x)
    assert( (snapshot == f(x)).all() )


# Compare interpolation to true value.
parameter = numpy.linspace(parameter[0], parameter[-1], 101)
interpolation = numpy.empty(parameter.shape)
for i, xi in enumerate(parameter):
    snapshot = model(xi)
    interpolation[i] = numpy.mean( snapshot )


plot.figure()
plot.plot(parameter, f(parameter), 'r-', label='true')
plot.plot(parameter, interpolation, 'k.', label='interpolation')
plot.xlabel('parameter')
plot.ylabel('model')
plot.legend()


# Compare backward interpolation to true inverse model.
g = lambda x: numpy.sqrt(9.1 - x)

obs = numpy.linspace(f(parameter[-1]), f(parameter[0]), 101)
parameter = numpy.empty(obs.shape)
for i, yi in enumerate(obs):
    inverse = model.invert(numpy.full(shape, yi))
    parameter[i] = numpy.mean(inverse)

plot.figure()
plot.plot(obs, g(obs), 'r-', label='true')
plot.plot(obs, parameter, 'k.', label='inv. interpolation')
plot.xlabel('observation')
plot.ylabel('parameter')
plot.legend()

plot.show()
