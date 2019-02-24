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

    x = df.loc[:, 'seed_cott_weight_(g)']
    y = df.loc[:, '2D_yield_area']

    if all:
        max_x = df.loc[:, 'seed_cott_weight_(g)'].max()
        max_y = df.loc[:, '2D_yield_area'].max()
    else:
        max_x = df.loc[df['year'] == year, 'seed_cott_weight_(g)'].max()
        max_y = df.loc[df['year'] == year, '2D_yield_area'].max()

    coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
    line_equation = clean_lin_eq(coeffs)

    x_linspace = np.linspace(0, max(x), len(x))
    y_hat = poly_eqn(x_linspace)

    ax.plot(x_linspace, y_hat, ':', color='#808B96', linewidth=3)
    ax.plot(x, y, 'o', color='#000000', markersize=8)

    plt.xlim(0 - int(max_x * .05), max_x + int(max_x * .10))
    plt.ylim(0 - int(max_y * .08), max_y + int(max_y * .13))

    left, right = plt.xlim()
    bottom, top = plt.ylim()

    function_position_x = left + right*.03
    function_position_y = top*.95 - (top*.12)

    r_square_position_x = left + right*.03
    r_square_position_y = top*.95 - (top*.06)

    r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

    ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 24})
    ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 26})

    ax.set_title(label=r"\[\textbf{UAV Yield vs Hand Harvested for All Samples and Flights\ " + str(year) + "}\]",
                fontdict={"fontsize": 20},
                pad=20)

    # Grams per meter square on the x axis
    # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)

    # Grams only on x axis
    ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    ax.set_ylabel(r"\[\textbf{Pixel Counts Corrected for Altitude}\ \left({cm}^{2}\right)\]", fontdict={"fontsize": 20},
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
    out_dir = '/home/will/uas-cotton-photogrammetry/output'

    # Create an out directory.
    directory_path = os.path.join(out_dir, "visuals_all")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    df = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/uas_yield_and_hand_harvested_all.csv')

    # Filtered.
    df_filt = df.loc[df['seed_cott_weight_(g)'] < 1500, :]

    point_filt =  [
        'id_11_2017.tif',
        'id_27_2017.tif',
        'id_1_2018.tif',
        'id_12_2018.tif',
        'id_20_2018.tif',
    ]

    df_filt = df_filt.loc[~df_filt['id_tag'].isin(point_filt), :]

    df_filt = df_filt.loc[df_filt['id_tag'] != 'id_20_2018.tif', :]
    df17_filt = df_filt.loc[df_filt['year'] == 2017, :]
    df18_filt = df_filt.loc[df_filt['year'] == 2018, :]

    # By year.
    df17 = df.loc[df['year'] == 2017, :]
    df18 = df.loc[df['year'] == 2018, :]

    # cols_alts_df17 = [
    #     ('#B3B6B7', 20),
    #     ('#909497', 22),
    #     ('#717D7E', 24),
    #     ('#616A6B', 26),
    #     ('#566573', 28),
    #     ('#000000', 30)
    # ]
    #
    # cols_alts_df18 = [
    #     ('#B3B6B7', 20),
    #     ('#909497', 35),
    #     ('#717D7E', 50),
    #     ('#616A6B', 75),
    #     ('#000000', 100),
    # ]


    all_out_both_filter = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_both_years_FILTERED.png")
    all_out_both = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_both_years.png")
    all_out_17_filter = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_year_2017_FILTERED.png")
    all_out_18_filter = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_year_2018_FILTERED.png")
    all_out_17 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_year_2017.png")
    all_out_18 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_year_2018.png")

    year_lingress(df=df_filt, out_path=all_out_both_filter, year='Both Years', all=True)
    year_lingress(df=df, out_path=all_out_both, year='Both Years', all=True)
    year_lingress(df=df17_filt, out_path=all_out_17_filter, year=2017)
    year_lingress(df=df18_filt, out_path=all_out_18_filter, year=2018)
    year_lingress(df=df17, out_path=all_out_17, year=2017)
    year_lingress(df=df18, out_path=all_out_18, year=2018)

