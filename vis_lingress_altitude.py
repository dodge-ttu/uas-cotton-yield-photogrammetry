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
    coefficients = ["{0:.3E}".format(i).split('E') for i in coefficients]
    coefficients.reverse()
    print(coefficients)
    pieces = []
    for ((cof1,cof2), deg) in zip(coefficients, degs):
        if deg == 0:
            if float(cof1) > 0:
                piece = "{0}{{E}}^{{{1}}}".format(cof1, cof2) if cof2 != '+00' else "{0}".format(cof1)
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


def altitude_lingress(df, cols_alts, out_path, year):

    max_x = df.loc[df['year'] == year, 'seed_cott_weight_(g)'].max()
    max_y = df.loc[df['year'] == year, '2D_yield_area'].max()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    slopes = []
    for (idx, (color, altitude)) in enumerate(cols_alts):
        x = df.loc[df['altitude'] == altitude, 'seed_cott_weight_(g)']
        y = df.loc[df['altitude'] == altitude, '2D_yield_area']

        coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
        line_equation = clean_lin_eq(coeffs)

        x_linspace = np.linspace(0, max(x), len(x))
        y_hat = poly_eqn(x_linspace)

        ax.plot(x_linspace, y_hat, ':', color=color, linewidth=3)
        ax.plot(x, y, 'o', color=color, markersize=8)

        plt.xlim(0 - int(max_x*.05), max_x + int(max_x*.10))
        plt.ylim(0 - int(max_y*.08), max_y + int(max_y*.13))

        left, right = plt.xlim()
        bottom, top = plt.ylim()

        altitude_position_x = left + right*.40
        altitude_position_y = top*.95 - (top*.043)*idx

        function_position_x = left + right*.15
        function_position_y = top*.95 - (top*.043)*idx

        r_square_position_x = left + right*.02
        r_square_position_y = top*.95 - (top*.043)*idx

        altitude_text = r"\[" + str(int(altitude)) + "{m}\]"
        r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

        ax.text(altitude_position_x, altitude_position_y, altitude_text, fontdict={"fontsize": 15}, color=color)
        ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 15}, color=color)
        ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 15}, color=color)

        slopes.append((coeffs[0], year, altitude))

    ax.set_title(label=r"\[\textbf{PCCA vs Hand Harvested by Altitude\ " + str(year) + "}\]",
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

    return slopes


def altitude_lingress_mean(df, out_path, year):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.gray()

    x = df.loc[:, 'seed_cott_weight_(g)'].groupby(df.loc[:, 'id_tag']).mean()
    y = df.loc[:, '2D_yield_area'].groupby(df.loc[:, 'id_tag']).mean()

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

    function_position_x = left + right * .03
    function_position_y = top * .95 - (top * .12)

    r_square_position_x = left + right * .03
    r_square_position_y = top * .95 - (top * .06)

    r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

    ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 24})
    ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 26})

    ax.set_title(label=r"\[\textbf{PCCA Mean Across Altitudes vs Hand Harvested\ " + str(year) + "}\]",
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

    ax.set_ylabel(r"\[\textbf{Pixel Counts Corrected for Altitude}\ \left({cm}^{2}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    plt.savefig(out_path)
    plt.close()


def altitude_multiples(df, out_path, h, w):

    altitudes = df.loc[:, 'altitude'].unique()
    altitudes = sorted(altitudes)

    fig, axs = plt.subplots(h,w, figsize=(20,10), sharex=True, sharey=True)
    plt.gray()

    for (altitude, ax) in zip(altitudes, axs.ravel()):
        x = df.loc[df['altitude'] == altitude, 'seed_cott_weight_(g)']
        y = df.loc[df['altitude'] == altitude, '2D_yield_area']

        coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
        line_equation = clean_lin_eq(coeffs)

        x_linspace = np.linspace(0, max(x), len(x))
        y_hat = poly_eqn(x_linspace)

        ax.plot(x_linspace, y_hat, '-', color="#8D8E8E")
        ax.plot(x, y, 'o', color="#000000")

        left, right = plt.xlim()
        bottom, top = plt.ylim()

        altx = right - right*.3
        alty = bottom + top*.3

        rsquarex = right - right*.3
        rsquarey = bottom + top*.2

        eqx = right - right*.3
        eqy = bottom + top*.1

        r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"
        altitude = r"\[Altitude\ " + str(int(altitude)) + "\]"
        year = df.loc[:, 'year'].values[0]

        ax.text(eqx, eqy, line_equation, fontdict={"fontsize": 9})
        ax.text(rsquarex, rsquarey, r_square, fontdict={"fontsize": 13})
        ax.text(altx, alty, altitude, fontdict={"fontsize": 13})

        ax.set_ylabel(r"\[\textbf{Pixel Counts Corrected for Altitude}\ \left({cm}^{2}\right)\]",
                      fontdict={"fontsize": 8},
                      labelpad=10)

        # Grams per meter square on the x axis
        # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
        #               fontdict={"fontsize": 20},
        #               labelpad=20)

        # Grams only on x axis
        ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\right)\]",
                      fontdict={"fontsize": 8},
                      labelpad=10)

        fig.suptitle(r"\[\textbf{PCCA vs Hand Harvested By Altitude\ " + str(year) + "}\]", fontdict={"fontsize": 8},
                     x=0.1, y=.95, horizontalalignment='left', verticalalignment='top')

    plt.savefig(out_path)
    plt.close()


def r_square_vs_altitude(df, out_path, year):


    altitudes = df.loc[:, 'altitude'].unique()
    altitudes = sorted(altitudes)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    r_square_values = []
    for altitude in altitudes:

        x = df.loc[df['altitude'] == altitude, 'seed_cott_weight_(g)']
        y = df.loc[df['altitude'] == altitude, '2D_yield_area']

        coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
        line_equation = clean_lin_eq(coeffs)

        r_square_values.append((r_square, altitude, year))

    x = altitudes
    y = [i[0] for i in r_square_values]

    print(x)
    print(y)

    coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
    line_equation = clean_lin_eq(coeffs)

    x_linspace = np.linspace(0, max(x), len(x))
    y_hat = poly_eqn(x_linspace)

    ax.plot(x_linspace, y_hat, ':', color='#808B96', linewidth=3)
    ax.plot(x, y, 'o', color='#000000', markersize=8)

    plt.xlim(-10,110)
    plt.ylim(.3,1)

    # function_position_x = 0
    # function_position_y = .4

    r_square_position_x = 0
    r_square_position_y = .5

    r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

    # ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 24})
    ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 26})

    ax.set_title(label=r"\[\textbf{R Square Values vs Altitude\ " + str(year) + "}\]",
                fontdict={"fontsize": 20},
                pad=20)

    # Grams per meter square on the x axis
    # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)

    # Grams only on x axis
    ax.set_xlabel(r"\[\textbf{Altitude}\ \left({m}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    ax.set_ylabel(r"\[\textbf{R}^\textbf{2}\]", fontdict={"fontsize": 20},
                labelpad=20)

    plt.savefig(out_path)
    plt.close()


def plot_slopes_from_lingress(slopes, year, out_path):
    slope = [i[0] for i in slopes]
    altitude = [i[2] for i in slopes]

    x = altitude
    y = slope

    print(max(slopes))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    coeffs, poly_eqn, r_square = get_poly_hat(x, y, 2)
    line_equation = clean_poly_eq(coeffs)

    x_linspace = np.linspace(0, max(x), len(x))
    y_hat = poly_eqn(x_linspace)

    ax.plot(x_linspace, y_hat, ':', color='#808B96', linewidth=3)
    ax.plot(x, y, 'o', color='#000000', markersize=8)

    plt.xlim(-10,110)
    plt.ylim(0, 6)

    function_position_x = 0
    function_position_y = 0.25

    r_square_position_x = 0
    r_square_position_y = 0.5

    r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

    ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 16})
    ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 18})

    ax.set_title(label=r"\[\textbf{Linear Model Slope vs Altitude\]",
                 fontdict={"fontsize": 20},
                 pad=20)

    # Grams per meter square on the x axis
    # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)

    # Grams only on x axis
    ax.set_xlabel(r"\[\textbf{Altitude}\ \left({m}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    ax.set_ylabel(r"\[\textbf{Slope}\]", fontdict={"fontsize": 20},
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

    df_all = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/uas_yield_and_hand_harvested_all.csv')

    # Filtered.
    df_filt = df_all.loc[df_all['seed_cott_weight_(g)'] < 1500, :]

    point_filt = [
        'id_11_2017.tif',
        'id_27_2017.tif',
        'id_1_2018.tif',
        'id_12_2018.tif',
        'id_20_2018.tif',
    ]

    df_filt = df_filt.loc[~df_filt['id_tag'].isin(point_filt), :]

    df_filt.to_csv('/home/will/uas-cotton-photogrammetry/output/uas_yield_and_hand_harvested_all_FILTERED.csv', index=False)

    df17_filt = df_filt.loc[df_filt['year'] == 2017, :]
    df18_filt = df_filt.loc[df_filt['year'] == 2018, :]

    # By year.
    df17 = df_all.loc[df_all['year'] == 2017, :]
    df18 = df_all.loc[df_all['year'] == 2018, :]

    cols_alts_df17 = [
        ('darkred', 20),
        ('darkblue', 22),
        ('darkviolet', 24),
        ('darkgreen', 26),
        ('darkorange', 28),
        ('crimson', 30)
    ]

    cols_alts_df18 = [
        ('darkred', 20),
        ('darkblue', 30),
        ('darkviolet', 35),
        ('darkgreen', 50),
        ('darkorange', 75),
        ('crimson', 100),
    ]

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

    all_out_17_filter = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_2017_FILTERED.png")
    all_out_18_filter = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_2018_FILTERED.png")
    all_out_17 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_2017.png")
    all_out_18 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_2018.png")

    mean_out_17_filter = os.path.join(directory_path,"UAV_Seeded_Cotton_Yield_Correlation_mean_altitude_2017_FILTERED.png")
    mean_out_18_filter = os.path.join(directory_path,"UAV_Seeded_Cotton_Yield_Correlation_mean_altitude_2018_FILTERED.png")
    mean_out_17 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_mean_altitude_2017.png")
    mean_out_18 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_mean_altitude_2018.png")

    mult_out_17_filter = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_multiples_2017_FILTERED.png")
    mult_out_18_filter = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_multiples_2018_FILTERED.png")
    mult_out_17 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_multiples_2017.png")
    mult_out_18 = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_Correlation_by_altitude_multiples_2018.png")

    rsquare_vs_alt_out = os.path.join(directory_path, "UAV_Seeded_Cotton_Yield_r_square_vs_altitude_both_years.png")

    slopes_17_filt = altitude_lingress(df=df17_filt, cols_alts=cols_alts_df17, out_path=all_out_17_filter, year=2017)
    slopes_18_filt = altitude_lingress(df=df18_filt, cols_alts=cols_alts_df18, out_path=all_out_18_filter, year=2018)
    slopes_17 = altitude_lingress(df=df17, cols_alts=cols_alts_df17, out_path=all_out_17, year=2017)
    slopes_18 = altitude_lingress(df=df18, cols_alts=cols_alts_df18, out_path=all_out_18, year=2018)

    altitude_lingress_mean(df=df17_filt, out_path=mean_out_17_filter, year=2017)
    altitude_lingress_mean(df=df18_filt, out_path=mean_out_18_filter, year=2018)
    altitude_lingress_mean(df=df17, out_path=mean_out_17, year=2017)
    altitude_lingress_mean(df=df18, out_path=mean_out_18, year=2018)

    altitude_multiples(df17_filt, out_path=mult_out_17_filter, h=2, w=3)
    altitude_multiples(df18_filt, out_path=mult_out_18_filter, h=2, w=3)
    altitude_multiples(df17, out_path=mult_out_17, h=2, w=3)
    altitude_multiples(df18, out_path=mult_out_18, h=2, w=3)

    r_square_vs_altitude(df=df_filt, out_path=rsquare_vs_alt_out, year='Both Years')

    slopes = slopes_17_filt + slopes_18_filt

    slopes_out = os.path.join(directory_path, "Linear_model_slopes_vs_altitude.png")
    print(slopes)
    print(slopes_18_filt)
    print(slopes_17_filt)

    plot_slopes_from_lingress(slopes=slopes, year='Both Years', out_path=slopes_out)

