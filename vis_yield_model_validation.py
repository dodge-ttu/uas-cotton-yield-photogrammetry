#!/home/will/uas-cotton-photogrammetry/cp-venv/bin/python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib import rc

# Must have latex installed.
def clean_poly_eq(coefficients, scientific_notation=True):
    n = len(coefficients)
    degs = list(range(n))
    coefficients = ["{0:.2E}".format(i).split('E') for i in coefficients]
    coefficients.reverse()
    pieces = []
    for ((cof1,cof2), deg) in zip(coefficients, degs):
        if deg == 0:
            if float(cof1) > 0:
                piece = "{0}{{E}}^{{{1}}}".format(cof1, cof2)
            else:
                piece = "{0}{{E}}^{{{1}}}".format(cof1, cof2)

        elif deg == 1:
            if float(cof1) > 0:
                piece = "+{0}{{E}}^{{{1}}}{{x}}".format(cof1, cof2)
            else:
                piece = "{0}{{E}}^{{{1}}}{{x}}".format(cof1, cof2)

        else:
            if float(cof1) > 0:
                piece = "+{0}{{E}}^{{{1}}}{{x}}^{{{2}}}".format(cof1, cof2, deg)
            else:
                piece = "{0}{{E}}^{{{1}}}{{x}}^{{{2}}}".format(cof1, cof2, deg)

        pieces.append(piece)

    pieces.reverse()

    equation = r"\[{y} = " + "".join(pieces[::-1]) + "\]"

    return equation

def clean_lin_eq(coefficients, scientific_notation=True):
    n = len(coefficients)
    coefficients = [round(i, 2) for i in coefficients]

    if coefficients[1] > 0:
        eqn_string = "{0}{{x}} + {1}".format(coefficients[0], coefficients[1])
    else:
        eqn_string = "{0}{{x}} {1}".format(coefficients[0], coefficients[1])

    equation = r"\[{y} = " + eqn_string + "\]"

    return equation

def get_poly_hat(x_values, y_values, poly_degree):
    coeffs = np.polyfit(x_values, y_values, poly_degree)
    poly_eqn = np.poly1d(coeffs)

    y_bar = np.sum(y_values) / len(y_values)
    ssreg = np.sum((poly_eqn(x_values) - y_bar) ** 2)
    sstot = np.sum((y_values - y_bar) ** 2)
    r_square = ssreg / sstot

    return (coeffs, poly_eqn, r_square)

def year_lingress(df, out_path, year, all=False):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.gray()

    x = df.loc[:, 'machine_harv_kg_per_ha'].values
    y = df.loc[:, 'predicted_kilo_per_ha'].values

    coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
    line_equation = clean_lin_eq(coeffs)

    x_linspace = np.linspace(0, max(x), len(x))
    y_hat = poly_eqn(x_linspace)

    # ax.plot(x_linspace, y_hat, ':', color='#808B96', linewidth=3)
    ax.plot(x, y, 'o', color='#000000', markersize=8)
    ax.plot([i for i in range(1800)], [i for i in range(1800)], ':', color='#808B96')

    # plt.xlim(0 - int(max_x * .05), max_x + int(max_x * .10))
    # plt.ylim(0 - int(max_y * .08), max_y + int(max_y * .13))

    plt.ylim(200,1800)
    plt.xlim(200,1800)

    left, right = plt.xlim()
    bottom, top = plt.ylim()

    #function_position_x = left + right*.03
    #function_position_y = top*.95 - (top*.12)

    r_square_position_x = left + right*.03
    r_square_position_y = top*.95 - (top*.06)

    r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

    #ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 24})
    ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 26})

    ax.set_title(label=r"\[\textbf{UAV Estimated Yield vs Machine Harvested Yield}\]",
                fontdict={"fontsize": 20},
                pad=20)

    # Grams per meter square on the x axis
    # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)

    # Grams only on x axis
    ax.set_xlabel(r"\[\textbf{Machine Harvested Yield}\ \left({kg}\cdot{ha}^{-1}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    ax.set_ylabel(r"\[\textbf{UAV Estimated Yield}\ \left({kg}\cdot{ha}^{-1}\right)\]", fontdict={"fontsize": 20},
                labelpad=20)

    plt.savefig(out_path)
    plt.close()


if __name__=="__main__":

    ## Use LaTex.
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    # Output directory.
    out_dir = '/home/will/uas-cotton-photogrammetry/output/'

    # Create an out directory.
    directory_path = os.path.join(out_dir, "visuals_all")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    df = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/yield_validation_counted_marked_2018/2018_valedation_pixel_counts_and_machine_harvested_yield.csv')

    df_out_validation = os.path.join(directory_path, "Predicted_vs_actual_yield_VALIDATION.png")

    year_lingress(df=df, out_path=df_out_validation, year='Both Years', all=True)