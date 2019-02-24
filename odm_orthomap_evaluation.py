import pandas as pd
import numpy as np
from matplotlib import rc
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.image as mpimg

from scipy.spatial import distance
from sklearn.neighbors import KernelDensity

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


def dist_base_vs_dist_gcp(df, out_path):

    flights = df.loc[:, 'flight_date'].unique()

    dfs = []
    for flight in flights:
        dfs.append(df.loc[df['flight_date'] == flight, :])

    x = df.loc[:, 'distance_to_nearest_gcp_m']
    y = df.loc[:, 'distance_to_base_m']*100 # scale to cm

    max_x = x.max()
    max_y = y.max()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    coeffs, poly_eqn, r_square = get_poly_hat(x, y, 1)
    line_equation = clean_lin_eq(coeffs)

    x_linspace = np.linspace(0, max(x), len(x))
    y_hat = poly_eqn(x_linspace)

    markers = ['P', '^', 'D']
    colors = ['#E74C3C', '#3498DB', '#16A085']

    sctrs = []
    for (df_scatter, marker, color) in zip(dfs, markers, colors):
        x_scatter = df_scatter.loc[:, 'distance_to_nearest_gcp_m'].values
        y_scatter = df_scatter.loc[:, 'distance_to_base_m']*100 # scale to cm
        aa = ax.scatter(x_scatter, y_scatter, c=color, marker=marker)
        sctrs.append(aa)

    ax.plot(x_linspace, y_hat, ':', color='#808B96', linewidth=3)
    ax.legend(handles=(sctrs[0], sctrs[1], sctrs[2]),
              labels=(flights[0], flights[1], flights[2]),
              loc=1,
              fontsize=18,
              framealpha=1.0,
    )
    plt.xlim(0 - int(max_x * .05), max_x + int(max_x * .10))
    plt.ylim(0 - int(max_y * .08), max_y + int(max_y * .13))

    left, right = plt.xlim()
    bottom, top = plt.ylim()

    # function_position_x = left + right * .03
    # function_position_y = top * .95 - (top * .12)

    r_square_position_x = left + right * .03
    r_square_position_y = top * .95 - (top * .06)

    r_square = r"\[{R}^{2}\ " + str(round(r_square, 3)) + "\]"

    # ax.text(function_position_x, function_position_y, line_equation, fontdict={"fontsize": 24})
    ax.text(r_square_position_x, r_square_position_y, r_square, fontdict={"fontsize": 26})

    ax.set_title(label=r"\[\textbf{Deviation From Base vs Distance to Nearest GCP}\]",
                 fontdict={"fontsize": 20},
                 pad=20)

    # Grams per meter square on the x axis
    # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)

    # Grams only on x axis
    ax.set_xlabel(r"\[\textbf{Distance To Nearest GCP}\ \left({m}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    ax.set_ylabel(r"\[\textbf{Deviation From Base Layer}\ \left({cm}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    plt.savefig(out_path)
    plt.close()


def dist_base_dist_gcp_interp(df, img, out_path, point_deviation_bounds):

    h, w, c = img.shape

    # lat = df.loc[:, 'lat_x'].values
    # lon = df.loc[:, 'lon_y'].values
    # z = df.loc[:, 'distance_to_base_m']

    lat = df.loc[:, 'lat_x'].values[:100]
    lon = df.loc[:, 'lon_x'].values[:100]

    # Get mean and convert to cm.
    z = df.loc[:, 'distance_to_base_m'].groupby(df.loc[:, 'point_id']).mean() * 100

    lat_min = point_deviation_bounds.loc[:, 'lat'].min()
    lat_max = point_deviation_bounds.loc[:, 'lat'].max()
    lon_min = point_deviation_bounds.loc[:, 'lon'].min()
    lon_max = point_deviation_bounds.loc[:, 'lon'].max()

    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min

    x_adjustment = w / lon_diff
    y_adjustment = h / lat_diff

    x = (lon - lon_min) * (x_adjustment * 1.0)
    y = (lat - lat_min) * (y_adjustment * 1.0)

    # Invert the values for the y axis because the image origin is 'upper left'.
    y = h - y

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    xi = np.linspace(0, w, w)
    yi = np.linspace(0, h, h)

    # Grid and interpolate the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    # Display image.
    ax.imshow(img, origin="upper")

    # # TriContour interpolation.
    # ax.tricontour(x, y, z, levels=20, colors='k', linewidths=[0.5], linestyles='--')
    # cntr1 = ax.tricontourf(x, y, z, levels=100, cmap="RdBu_r", alpha=0.5)
    # fig.colorbar(cntr1, ax=ax)

    # Linear interpolation.
    plt.contour(xi, yi, zi, 15, levels=20, colors='k', linewidths=[0.5], linestyles='--')
    plt.contourf(xi, yi, zi, 15, levels=100, cmap=plt.cm.jet, alpha=0.4)
    cbar = plt.colorbar()  # draw colorbar

    plt.tick_params(
        axis='both',
        which='both',
        labelbottom=False,
        labelleft=False,
    )

    cbar.ax.set_ylabel(r"\[\left({cm}\right)\]",
                    fontdict={"fontsize": 20}, labelpad=40, rotation=270)

    plt.scatter(x, y, marker='o', c='b', s=5)

    ax.set_title(label=r"\[\textbf{Interpolated Mean Deviation From Base Point}\]",
                 fontdict={"fontsize": 20},
                 pad=20)

    # Grams per meter square on the x axis
    # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)
    # Grams only on x axis
    # ax.set_xlabel(r"\[\textbf{Distance To Nearest GCP}\ \left({m}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)
    #
    # ax.set_ylabel(r"\[\textbf{Deviation From Base Point}\ \left({cm}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)

    plt.savefig(out_path)
    plt.close()


def histogram_all_deviations(df, out_path):

    np.random.seed(8675309)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    data = df.loc[:, 'distance_to_base_m'].values * 100 # convert to centimeters
    data = data[data < 15.6]
    bins = np.linspace(0, 20, 41)
    ax.hist(data, color='#0FC25B', edgecolor='k', bins=bins, rwidth=0.80, density=True, alpha=0.5)

    # Reshape data for scikit-learn.
    data = data[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)

    # Resize bins for probability density function.
    bins = np.linspace(0, 16, 301)[:, np.newaxis]
    log_dens = kde.score_samples(bins)

    ax.plot(bins[:, 0], np.exp(log_dens), 'r-')
    ax.plot(data[:, 0], -0.008 - 0.02 * np.random.random(data.shape[0]), 'kd')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(-.2, 16.2)
    ax.set_ylim(-0.035, 0.26)

    # Grams per meter square on the x axis
    # ax.set_xlabel(r"\[\textbf{Hand Harvested Yield}\ \left({g}\cdot{m}^{-2}\right)\]",
    #               fontdict={"fontsize": 20},
    #               labelpad=20)

    # Grams only on x axis

    ax.set_xlabel(r"\[\textbf{Deviation From Base Layer}\ \left({cm}\right)\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    ax.set_ylabel(r"\[\textbf{Density}\]",
                  fontdict={"fontsize": 20},
                  labelpad=20)

    ax.set_title(label=r"\[\textbf{Distribution of Deviation From Base Layer}\]",
                 fontdict={"fontsize": 20},
                 pad=20)

    plt.savefig(out_path)
    plt.close()


if __name__=='__main__':

    ## Use LaTex.
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    df_1_from_gcp = pd.read_csv('/home/will/uas-cotton-photogrammetry/dist_from_GCP_non_GCP_measure_1_2018-07-25_75_75_50_pivot_odm_orthophoto_modified.csv', header=[0])
    df_2_from_gcp = pd.read_csv('/home/will/uas-cotton-photogrammetry/dist_from_GCP_non_GCP_measure_2_2018-08-13_75_75_50_pivot_odm_orthophoto_modified.csv', header=[0])
    df_3_from_gcp = pd.read_csv('/home/will/uas-cotton-photogrammetry/dist_from_GCP_non_GCP_measure_3_2018-09-04_75_75_50_rainMatrix_odm_orthophoto_modified.csv', header=[0])

    df_1_from_base = pd.read_csv('/home/will/uas-cotton-photogrammetry/dist_from_base_non_GCP_measure_1_2018-07-25_75_75_50_pivot_odm_orthophoto_modified.csv', header=[0])
    df_2_from_base = pd.read_csv('/home/will/uas-cotton-photogrammetry/dist_from_base_non_GCP_measure_2_2018-08-13_75_75_50_pivot_odm_orthophoto_modified.csv', header=[0])
    df_3_from_base = pd.read_csv('/home/will/uas-cotton-photogrammetry/dist_from_base_non_GCP_measure_3_2018-09-04_75_75_50_rainMatrix_odm_orthophoto_modified.csv', header=[0])

    point_deviation_bounds = pd.read_csv('/home/will/uas-cotton-photogrammetry/point_deviation_bounding_box.csv', header=[0])

    dfs_from_gcp = [
        df_1_from_gcp,
        df_2_from_gcp,
        df_3_from_gcp,
    ]

    dfs_from_base = [
        df_1_from_base,
        df_2_from_base,
        df_3_from_base,
    ]

    col_names_from_gcp = [
        'lat',
        'lon',
        'point_id',
        'gcp_id',
        'distance_to_nearest_gcp_m',
    ]

    col_names_from_base = [
        'lat',
        'lon',
        'point_id',
        'base_point_id',
        'distance_to_base_m',
    ]

    flight_one = '2018-07-25'
    flight_two = '2018-08-13'
    flight_three = '2018-09-04'

    flight_dates = [flight_one, flight_two, flight_three]

    for (flight_date, df) in zip(flight_dates, dfs_from_gcp):
        df.columns = col_names_from_gcp
        df.loc[:, 'flight_date'] = flight_date

    for df in dfs_from_base:
        df.columns = col_names_from_base

    dfs_merged = []
    counter = 0
    for (df_base, df_gcp) in zip(dfs_from_base, dfs_from_gcp):
        counter += 1
        df = df_base.merge(df_gcp, on='point_id', how='outer')
        dfs_merged.append(df)

    df_all = pd.concat(dfs_merged)

    dist_base_dist_gcp_outpath = '/home/will/uas-cotton-photogrammetry/output/visuals_all/dist_from_base_vs_dist_to_nearest_gcp.png'

    dist_base_vs_dist_gcp(df_all, out_path=dist_base_dist_gcp_outpath)

    img_path = '/home/will/uas-cotton-photogrammetry/orthophoto_deviation_evaluation.tif'
    img = mpimg.imread(img_path)

    dist_interp_outpath = '/home/will/uas-cotton-photogrammetry/output/visuals_all/orthophoto_deviation_interpolation.png'

    dist_base_dist_gcp_interp(df=df_all, img=img, out_path=dist_interp_outpath, point_deviation_bounds=point_deviation_bounds)

    deviation_histograms = '/home/will/uas-cotton-photogrammetry/output/visuals_all/orthophoto_deviation_histogram.png'

    histogram_all_deviations(df=df_all, out_path=deviation_histograms)









    # lat_min = point_deviation_bounds.loc[:, 'lat'].min()
    # lat_max = point_deviation_bounds.loc[:, 'lat'].max()
    # lon_min = point_deviation_bounds.loc[:, 'lon'].min()
    # lon_max = point_deviation_bounds.loc[:, 'lon'].max()
    #
    # lat = df_all.loc[:, 'lat_x'].values
    # lon = df_all.loc[:, 'lon_x'].values
    #
    # # lat = dfs_merged[0].loc[:, 'lat_x'].values
    # # lon = dfs_merged[0].loc[:, 'lon_x'].values
    #
    # df_gcp = pd.read_csv('/home/will/uas-cotton-photogrammetry/gcp_locations.csv', header=[0])
    # df_gcp.columns = ['lat', 'lon', 'id', 'nada']
    #
    # x = lon
    # y = lat
    # x_gcp = df_gcp.loc[:, 'lon'].values
    # y_gcp = df_gcp.loc[:, 'lat'].values
    # z = df_all.loc[:, 'distance_to_base_m'].values
    #
    # fig, ax = plt.subplots(1,1, figsize=(8, 20))
    #
    # ax.tricontour(x, y, z, levels=10, colors='k', linewidths=[0.5], linestyles='--')
    # cntr1 = ax.tricontourf(x, y, z, levels=100, cmap="RdBu_r")
    #
    # fig.colorbar(cntr1, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 20))
    #
    # # Bottom plot.
    # triang = tri.Triangulation(x, y)
    # refiner = tri.UniformTriRefiner(triang)
    # tri_refi, z_test_refi = refiner.refine_field(z, subdiv=4)
    # ax.tricontour(tri_refi, z_test_refi, levels=20, colors='k', linewidths=[0.5])
    # cntr2 = ax.tricontourf(tri_refi, z_test_refi, levels=20, cmap="RdBu_r")
    # fig.colorbar(cntr2, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)




