#!/usr/bin/env python3

import sys
sys.path.append('share')

import argparse
import mim
import mimData
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def getData(path):
    return mimData.AnalysisData.load(path)

def createModel(paths, acceptance, time):
    fluxes = [getData(path) for path in paths]    
    parameter = np.array([flux.data.density for flux in fluxes])
    images = np.stack(
        [flux.data.mean * acceptance.mean * time * 30 * 24 * 3600 for flux in fluxes]
    )
    
    return mim.Model(parameter, images)


def plotImage(x, y, z, path, ymax=25,
        vmin=0.0, vmax=1.0, log_scale=False):
    return
    
    plt.figure(figsize=(16, 7))
    
    if log_scale == True:
        plt.pcolormesh(x, y, z,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap=plt.cm.jet, shading='auto')
    else:
        plt.pcolormesh(x, y, z,
                vmin=vmin, vmax=vmax,
                cmap=plt.cm.jet, shading='auto')
    
    plt.xlabel('azimut [deg]')
    plt.ylabel('elevation [deg]')
    if ymax > 0:
        plt.ylim(0, ymax)
    plt.colorbar()
    plt.savefig(path)


def treatData(args):
    # Pseudo random number generator
    prng = mim.Prng()
    random = mimData.Random(prng)
    
    # Observation flux
    flux0 = getData(args.observation).data    
    plotImage(flux0.azimuths, flux0.elevations, flux0.mean.T,
            f"share/plots/flux-density-{flux0.density}.png",
            ymax=25, vmin=1E-13, vmax=1E-03, log_scale=True)
    
    
    # Acceptance
    acceptance = getData(args.acceptance).data    
    plotImage(acceptance.azimuths, acceptance.elevations, acceptance.mean.T,
            f"share/plots/acceptance.png",
            ymax=-1, vmin=0.0, vmax=1.0, log_scale=False)


    # Filter
    filter_ = getData(args.filter).data    
    plotImage(filter_.azimuths, filter_.elevations, filter_.mean.T,
            f"share/plots/filter.png",
            ymax=25, vmin=-1.0, vmax=1.0, log_scale=False)


    # Observation rate
    obs = flux0.mean * acceptance.mean * args.time * 30 * 24 * 3600
    plotImage(flux0.azimuths, flux0.elevations,
            np.ma.masked_where(filter_.mean < 0, obs, copy=True).T,
            f"share/plots/rate-months-{args.time}.png",
            ymax=25, vmin=1E-07, vmax=1E+04, log_scale=True)

    
    # Model
    model = createModel((path for path in args.files), acceptance, args.time)
    print(f"shape  = {model.shape}")
    print(f"pmin   = {model.pmin}")
    print(f"pmax   = {model.pmax}")    
    print(f"mvalue = {args.mvalue}")
    print(f"sigma  = {args.sigma}")
    print(f"time   = {args.time}")
    print(f"par    = {flux0.density}")
    
    # Compute stats
    image_mean = np.zeros(obs.shape)
    image_mean2 = np.zeros(obs.shape)
    N = args.events
    for i in range(args.events):
        print(f"process {i+1}/{N}...")
        # Get a random observation from Poisson distribution
        robs = random.randomize(obs)
        # Invert the observation
        image, _, _ = model.invert_min(robs,
                args.mvalue, filter_.mean, args.sigma)
        #image = np.ma.masked_where(filter_.mean < 0, image, copy=False)
        
        image_mean += image
        image_mean2 += image * image

    image_mean /= N
    image_rms = np.sqrt((image_mean2/N - image_mean*image_mean) / N)
    image_rms /= image_mean
    image_rms *= 100
    
    
    # Plot stats
    plotImage(flux0.azimuths, flux0.elevations,
            np.ma.masked_where(filter_.mean < 0, image_mean, copy=False).T,
            f"share/plots/mean-density-{flux0.density}-time-{args.time}-mvalue-{args.mvalue}.png",
            ymax=25, vmin=model.pmin, vmax=model.pmax, log_scale=False)
    
    plotImage(flux0.azimuths, flux0.elevations,
            np.ma.masked_where(filter_.mean < 0, image_rms, copy=False).T,
            f"share/plots/rms-density-{flux0.density}-time-{args.time}-mvalue-{args.mvalue}.png",
            ymax=25, vmin=1E-02, vmax=5E+01, log_scale=True)
    
    # Save image
    data = mimData.AnalysisData(image_mean)
    data.dump("share/mim/data/image_mean-{flux0.density}-time-{args.time}-mvalue-{args.mvalue}.pkl.gz")
    data = mimData.AnalysisData(image_rms)
    data.dump("share/mim/data/image_rms-{flux0.density}-time-{args.time}-mvalue-{args.mvalue}.pkl.gz")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute statistics on inverting data")
    
    parser.add_argument("-f",
        dest="files",
        help="input filename",
        nargs="+")
    
    parser.add_argument("-o",
        dest = "observation",
        help="hypothesis observation")
    
    parser.add_argument("-a",
        dest = "acceptance",
        help="acceptance")
    
    parser.add_argument("-l",
        dest = "filter",
        help="filter")
    
    parser.add_argument("-t",
        dest = "time",
        help="duration of data taking in [months]",
        default=12.0,
        type=float)
    
    parser.add_argument("-m",
        dest = "mvalue",
        help="the minimum value",
        default=10.0,
        type=float)
    
    parser.add_argument("-s",
        dest = "sigma",
        help="the gaussian sigma",
        default=1.0,
        type=float)
    
    parser.add_argument("-n",
        dest = "events",
        help="the number of random observations",
        default=300,
        type=int)    
    
    args = parser.parse_args()
    
    treatData(args)
