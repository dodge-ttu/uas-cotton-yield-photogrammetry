#!/home/will/uas-cotton-photogrammetry/cp-venv/bin/python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib import rc
from equation_plotting_helpers_latex import get_poly_hat, clean_poly_eq


def altitude_lingress(df, cols_alts, out_path, year):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for (idx, (color, altitude)) in enumerate(cols_alts):
        x = df.loc[df['altitude'] == altitude, '2D_yield_area']
        y = df.loc[df['altitude'] == altitude, 'seed_cott_weight_(g)']

        coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
        line_equation = clean_poly_eq(coeffs)

        x_linspace = np.linspace(0, max(x), len(x))
        y_hat = poly_eqn(x_linspace)

        ax.plot(x_linspace, y_hat, '-', color=color)
        ax.plot(x, y, 'o', color=color)

        function_position_x = (x_linspace[-1]) + 50
        function_position_y = (y_hat[-1]) + 50

        r_square_position_x = (x_linspace[-1]) + 100
        r_square_position_y = (y_hat[-1]) + 100

        r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

        ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 14}, color=color)
        ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 18}, color=color)

        ax.set_title(label=r"\[\textbf{UAV Seeded Cotton Yield Measurement" + str(year) + "}\]",
                     fontdict={"fontsize": 20},
                     pad=20)
        ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]", fontdict={"fontsize": 20},
                      labelpad=20)
        ax.set_ylabel(r"\[\textbf{UAV Measured Seeded Cotton}\ \left({cm}^{2}\right)\]", fontdict={"fontsize": 20},
                      labelpad=20)

        plt.savefig(out_path)

def altitude_multiples(df, out_path, h, w, rsquarex, rsquarey, eqx, eqy, altx, alty):

    altitudes = df.loc[:, 'altitude'].unique()

    fig, axs = plt.subplots(h,w, figsize=(20,10), sharex=True, sharey=True)
    plt.gray()

    for (altitude, ax) in zip(altitudes, axs.ravel()):
        print(altitude)
        x = df.loc[df['altitude'] == altitude, '2D_yield_area']
        y = df.loc[df['altitude'] == altitude, 'seed_cott_weight_(g)']

        coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
        line_equation = clean_poly_eq(coeffs)

        x_linspace = np.linspace(0, max(x), len(x))
        y_hat = poly_eqn(x_linspace)

        ax.plot(x_linspace, y_hat, '-', color="#8D8E8E")
        ax.plot(x, y, 'o', color="#000000")

        function_position_x = (eqx)
        function_position_y = (eqx)

        r_square_position_x = (rsquarex)
        r_square_position_y = (rsquarey)

        r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"
        altitude = r"\[Altitude\ " + str(int(altitude)) + "\]"
        year = df.loc[:, 'year'].values[0]

        ax.text(eqx, eqy, line_equation, fontdict={"fontsize": 8})
        ax.text(rsquarex, rsquarey, r_square, fontdict={"fontsize": 12})
        ax.text(altx, alty, altitude, fontdict={"fontsize": 12})


        # ax.set_title(label=r"\[\textbf{UAV Seeded Cotton Yield Measurement" + str(year) + "}\]",
        #              fontdict={"fontsize": 20},
        #              pad=20)

        ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]", fontdict={"fontsize": 8},
                      labelpad=20)
        ax.set_ylabel(r"\[\textbf{UAV Measured Seeded Cotton}\ \left({cm}^{2}\right)\]", fontdict={"fontsize": 8},
                      labelpad=20)

        fig.suptitle(r"\[\textbf{UAV Seeded Cotton Yield Measurement" + str(year) + "}\]", fontdict={"fontsize": 8})

        plt.savefig(out_path)


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

    df17 = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2017/2017_pixel_counts_and_hand_harvested_yield.csv')
    df18 = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2018/2018_pixel_counts_and_hand_harvested_yield.csv')

    df17 = df17.loc[df17['seed_cott_weight_(g)'] < 1500, :]
    df18 = df18.loc[df18['seed_cott_weight_(g)'] < 1500, :]


    cols_alts_df17 = [
        ('red', 20),
        ('blue', 22),
        ('violet', 24),
        ('green', 26),
        ('darkorange', 28),
        ('crimson', 30)
    ]

    cols_alts_df18 = [
        ('red', 20),
        ('blue', 35),
        ('violet', 50),
        ('green', 75),
        ('darkorange', 100),
    ]

    out_path_17 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_2017.png")
    out_path_18 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_2018.png")

    altitude_lingress(df=df17, cols_alts=cols_alts_df17, out_path=out_path_17, year=2017)
    altitude_lingress(df=df18, cols_alts=cols_alts_df18, out_path=out_path_18, year=2018)
    altitude_multiples(df17, out_path_17, h=2, w=3, eqx=2000, eqy=200, rsquarex=2000, rsquarey=325, altx=2000, alty=450)
    altitude_multiples(df18, out_path_18, h=2, w=3, eqx=2500, eqy=200, rsquarex=2500, rsquarey=350, altx=2500, alty=500)

