#!/home/will/uas-cotton-photogrammetry/cp-venv/bin/python

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib import rc
from equation_plotting_helpers_latex import get_poly_hat, clean_poly_eq

if __name__=="__main__":

    ## Use LaTex.
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    # Output directory.
    out_dir = '/home/will/uas-cotton-photogrammetry/output'

    # Create an out directory.
    directory_path = os.path.join(out_dir, "visuals_2018")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    df = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2018/2018_pixel_counts_and_hand_harvested_yield.csv')

    x = df.loc[:, '2D_yield_area']
    y = df.loc[:, 'seed_cott_weight_(g)']
    tag = df.loc[:, 'altitude']

    coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
    line_equation = clean_poly_eq(coeffs)

    x_linspace = np.linspace(0, max(x), len(x))
    y_hat = poly_eqn(x_linspace)

    N=10

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    norm = mpl.colors.BoundaryNorm([0,20,35,50,75,101], cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(x_linspace, y_hat, '-', color="#8D8E8E")
    ax.scatter(x, y, c=tag, cmap=cmap, norm=norm)

    ax.set_title(label=r"\[\textbf{UAV Seeded Cotton Yield Measurement Model 2018}\]", fontdict={"fontsize":20}, pad=20)
    ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]", fontdict={"fontsize":20}, labelpad=20)
    ax.set_ylabel(r"\[\textbf{UAV Measured Seeded Cotton}\ \left({cm}^{2}\right)\]", fontdict={"fontsize":20}, labelpad=20)

    function_position_x = max(x)-(.70*max(x))
    function_position_y = max(y)-(.98*max(y))

    r_square_position_x = max(x) - (.95 * max(x))
    r_square_position_y = max(y) - (.10 * max(y))

    r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

    ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize":14})
    ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize":18})

    plt.savefig(os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Measurement_Model_2018.png"))

