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


def plotImage(x, y, z, path, ymax=25,
        vmin=0.0, vmax=1.0, log_scale=False):
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


def plotData(args):
    # Observation flux
    flux0 = getData(args["observation"]).data    
    plotImage(flux0.azimuths, flux0.elevations, flux0.mean.T,
            f"share/plots/flux-density-{flux0.density}.png",
            ymax=25, vmin=1E-13, vmax=1E-03, log_scale=True)
    
    # Acceptance
    acceptance = getData(args["acceptance"]).data    
    plotImage(acceptance.azimuths, acceptance.elevations, acceptance.mean.T,
            f"share/plots/acceptance.png",
            ymax=-1, vmin=0.0, vmax=1.0, log_scale=False)


    # Filter
    filter_ = getData(args["filter"]).data
    plotImage(filter_.azimuths, filter_.elevations, filter_.mean.T,
            f"share/plots/filter.png",
            ymax=25, vmin=-1.0, vmax=1.0, log_scale=False)

    """
    # Observation rate
    obs = getData(args["rate"]).data
    plotImage(flux0.azimuths, flux0.elevations,
            np.ma.masked_where(filter_.mean < 0, obs, copy=True).T,
            f"share/plots/rate-months-{args.time}.png",
            ymax=25, vmin=1E-07, vmax=1E+04, log_scale=True)

    """
    
    # Plot stats
    image_mean = getData(args["image_mean"]).data
    plotImage(flux0.azimuths, flux0.elevations,
            np.ma.masked_where(filter_.mean < 0, image_mean, copy=False).T,
            f"share/plots/mean-density.png",
            ymax=25, vmin=1.0, vmax=3.0, log_scale=False)
    
    image_rms = getData(args["image_rms"]).data
    plotImage(flux0.azimuths, flux0.elevations,
            np.ma.masked_where(filter_.mean < 0, image_rms, copy=False).T,
            f"share/plots/rms-density.png",
            ymax=25, vmin=1E-02, vmax=5E+01, log_scale=True)

    plt.figure(figsize=(16, 7))
    plt.hist(image_rms, bins=50)
    #plt.xlim(0, 5)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("share/plots/rms-density-dist.png")


if __name__ == "__main__":
    args = {
        "observation":"share/mim/data/flux-1.80.pkl.gz",
        "acceptance": "share/mim/data/acceptance.pkl.gz",
        "filter": "share/mim/data/filter.pkl.gz",
        "image_mean": "share/mim/data/image_mean.pkl.gz",
        "image_rms": "share/mim/data/image_rms.pkl.gz",
    }
    
    plotData(args)
