import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.image as mpimg
from matplotlib import rc


def pixel_intensity_comaprison(flat_imgs, resized_imgs, out_path, skip_n=20, grid_interval=5):
    # Plot sample site images for each year and a sparkline of the flattened blue channel.
    fig, axs = plt.subplots(6, 2, figsize=(12, 12), gridspec_kw={'width_ratios': [1, 3]})
    # plt.gray()
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    for idx, (flat_img, (img, altitude, year)) in enumerate(zip(flat_imgs, resized_imgs)):
        b, g, r = cv2.split(img)
        h, w, c = img.shape
        axs[idx, 0].imshow(g)

        # Major ticks every 20, minor ticks every 5
        major_ticks_x = np.arange(0, w, grid_interval)
        major_ticks_y = np.arange(0, h, grid_interval)

        axs[idx, 0].set_xticks(major_ticks_x, minor=True)
        axs[idx, 0].set_yticks(major_ticks_y, minor=True)

        axs[0][0].set_title(r"\[\textbf{Scaled Sample Images}\]",
                            fontdict={"fontsize": 12}, pad=10)

        # And a corresponding grid
        axs[idx, 0].grid(which='minor', axis='both', linestyle='-', color='#2AFF00', linewidth=.2)

        # Plot sparklines.
        x = np.array(range(len(flat_img[flat_img > 0][::skip_n])))
        y = flat_img[flat_img > 0][::skip_n]

        axs[idx, 1].plot(x, y, '+')

        axs[idx, 1].plot([x[5], x[5]], [0, y[5]], 'r--')
        axs[idx, 1].plot([x[10], x[10]], [0, y[10]], 'r--')
        axs[idx, 1].plot([x[15], x[15]], [0, y[15]], 'r--')
        axs[idx, 1].plot([x[20], x[20]], [0, y[20]], 'r--')
        axs[idx, 1].plot([x[25], x[25]], [0, y[25]], 'r--')
        axs[idx, 1].plot([x[30], x[30]], [0, y[30]], 'r--')
        #axs[idx, 1].plot([x[35], x[35]], [0, y[35]], 'r-')

        axs[idx, 1].set_xlim(0, 35)
        axs[idx, 1].set_ylim(0, 265)

        altitude = r"\[" + str(altitude) + "\ {m}\]"

        axs[idx, 1].text(1, 220, altitude, fontdict={"fontsize": 12})

        axs[idx, 1].set_ylabel(r"\[\textbf{Blue Channel Value}\]",
                      fontdict={"fontsize": 8},
                      labelpad=10)

        axs[idx, 1].set_xlabel(r"\[\textbf{Pixel Sample ID}\]",
                      fontdict={"fontsize": 12},
                      labelpad=10)

        axs[0][1].set_title(r"\[\textbf{Blue Channel Value Comparison by Sample ID}\]",
                               fontdict={"fontsize": 12}, pad=10)

    fig.suptitle(r"\[\textbf{Comparison of Pixel Intensities}\]", fontsize=24,
                horizontalalignment='center', verticalalignment='top')

    plt.savefig(out_path)
    plt.close()


if __name__=="__main__":

    ## Use LaTex.
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    # Samples from 2017.
    samp_10_20m_2017 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/2017-11-17_75_75_20_odm_orthophoto_modified-p1-extracted/id_10_2017.tif'
    samp_10_22m_2017 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/2017-11-17_75_75_22_odm_orthophoto_modified-p1-extracted/id_10_2017.tif'
    samp_10_24m_2017 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/2017-11-16_75_75_24_odm_orthophoto_modified-p1-extracted/id_10_2017.tif'
    samp_10_26m_2017 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/2017-11-16_75_75_26_odm_orthophoto_modified-p1-extracted/id_10_2017.tif'
    samp_10_28m_2017 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/2017-11-16_75_75_28_odm_orthophoto_modified-p1-extracted/id_10_2017.tif'
    samp_10_30m_2017 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/2017-11-16_75_75_30_odm_orthophoto_modified-p1-extracted/id_10_2017.tif'

    # Samples from 2018.
    samp_12_20m_2018 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/2018-10-23_65_75_20_rainMatrix_odm_orthophoto_modified-p1-extracted/id_17_2018.tif'
    samp_12_30m_2018 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/2018-10-26_65_75_30_rainMatrix_odm_orthophoto_modified-p1-extracted/id_17_2018.tif'
    samp_12_35m_2018 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/2018-10-23_65_75_35_rainMatrix_odm_orthophoto_modified-p1-extracted/id_17_2018.tif'
    samp_12_50m_2018 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/2018-10-23_65_75_50_rainMatrix_odm_orthophoto_modified-p1-extracted/id_17_2018.tif'
    samp_12_75m_2018 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/2018-10-23_65_75_75_rainMatrix_odm_orthophoto_modified-p1-extracted/id_17_2018.tif'
    samp_12_100m_2018 = '/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/2018-10-23_65_75_100_rainMatrix_odm_orthophoto_modified-p1-extracted/id_17_2018.tif'

    # Out directory.
    out_path = '/home/will/uas-cotton-photogrammetry/output/resized_pixel_consistency_samples'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    samps_2017 = [
        (samp_10_20m_2017, 20, 2017),
        (samp_10_22m_2017, 22, 2017),
        (samp_10_24m_2017, 24, 2017),
        (samp_10_26m_2017, 26, 2017),
        (samp_10_28m_2017, 28, 2017),
        (samp_10_30m_2017, 30, 2017),

    ]

    samps_2018 = [
        (samp_12_20m_2018, 20, 2018),
        (samp_12_30m_2018, 30, 2018),
        (samp_12_35m_2018, 35, 2018),
        (samp_12_50m_2018, 50, 2018),
        (samp_12_75m_2018, 75, 2018),
        (samp_12_100m_2018, 100, 2018),
    ]

    imgs_2017 = [(cv2.imread(i[0]), i[1], i[2]) for i in samps_2017]
    imgs_2018 = [(cv2.imread(i[0]), i[1], i[2]) for i in samps_2018]

    img_sizes_2017 = [i[0].shape for i in imgs_2017]
    img_sizes_2018 = [i[0].shape for i in imgs_2018]

    smallest_2017 = min(img_sizes_2017, key=lambda x: x[0])
    smallest_2018 = min(img_sizes_2018, key=lambda x: x[0])

    # Resize images so the pixel sampling is uniform.
    resized_2017 = []
    for (img, altitude, year) in imgs_2017:
        h, w, c = img.shape
        # r = smallest_2017[0] / h
        # dim = (int(h * r), int(w * r))
        dim = (smallest_2017[0], smallest_2017[1])
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_2017.append((resized, altitude, year))

    resized_2018 = []
    for (img, altitude, year) in imgs_2018:
        h, w, c = img.shape
        # r = smallest_2017[0] / h
        # dim = (int(h * r), int(w * r))
        dim = (smallest_2018[0], smallest_2018[1])
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_2018.append((resized, altitude, year))

    # Write resized samples.
    for (idx, (img, altitude, year)) in enumerate(resized_2017):
        file_path = os.path.join(out_path, '{0}_{1}_{2}_resized.tif'.format(year, idx, altitude))
        cv2.imwrite(file_path, img)

    for (idx, (img, altitude, year)) in enumerate(resized_2018):
        file_path = os.path.join(out_path, '{0}_{1}_{2}_resized.tif'.format(year, idx, altitude))
        cv2.imwrite(file_path, img)


    # Flatten resized images for both years.
    flat_imgs_2017 = []
    for (img, altitude, year) in resized_2017:
        b,g,r = cv2.split(img)
        flat = b.ravel()
        flat_imgs_2017.append(flat)

    flat_imgs_2018 = []
    for (img, altitude, year) in resized_2018:
        b,g,r = cv2.split(img)
        flat = b.ravel()
        flat_imgs_2018.append(flat)

    pixel_intens_compare_out_2017 = '/home/will/uas-cotton-photogrammetry/output/visuals_all/pixel_intesity_comparisons_2017.png'
    pixel_intens_compare_out_2018 = '/home/will/uas-cotton-photogrammetry/output/visuals_all/pixel_intesity_comparisons_2018.png'

    pixel_intensity_comaprison(flat_imgs_2017, resized_2017, pixel_intens_compare_out_2017, skip_n=300, grid_interval=6)
    pixel_intensity_comaprison(flat_imgs_2018, resized_2018, pixel_intens_compare_out_2018, skip_n=50, grid_interval=3.5)
