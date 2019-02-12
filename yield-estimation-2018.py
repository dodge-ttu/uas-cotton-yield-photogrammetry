import os
import re
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def count_white_pix(sample_images=None, thresh_value=None):

    font = cv2.FONT_HERSHEY_SIMPLEX

    images_counted_marked = []
    pixel_counts = []
    masks = []

    for image, ID_tag in sample_images:
        image_copy = image.copy()
        h, w, c = image.shape
        b, g, r = cv2.split(image)

        img_gray = b
        #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #img_gray = cv2.medianBlur(img_gray, 7)
        (T, thresh) = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)

        mask_data = np.nonzero(thresh)

        x, y = mask_data
        x = x.tolist()
        y = y.tolist()
        marks = [(x, y) for (x, y) in zip(x, y)]

        pixel_counts.append((len(marks), ID_tag))

        for i in marks:
            # cv2.circle(img, (i[1], i[0]), 1, (255,255,255), 1)
            image_copy[i] = (0, 0, 255)

        images_counted_marked.append((image_copy, ID_tag))
        masks.append((mask_data, ID_tag, (h, w)))

    return images_counted_marked, pixel_counts, masks


if __name__ == "__main__":

    # Details.
    plantings = ['earlier', 'later']
    what = 'aoms'

    # Define path to output directory.
    output_dir = "/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2018/"

    # Provide an ID for the analysis.
    input_layer_name_earlier = "2018-10-26_65_75_30_rainMatrix_odm_orthophoto_modified"  # 100 pixels / meter
    input_layer_name_later = "2018-11-15_65_75_35_rainMatrix_modified"  # 90.9091 pixels / meter

    # The GSD values are given to Open Drone Map in the orthophoto creation phase. These values are then stored.
    # This is why, as in the case above, the flight with lower altitude has a courser per-pixel resolution.
    # With another orthophoto processing the values can be altered.

    layer_ids = [input_layer_name_earlier, input_layer_name_later]

    # GSD values for each input layer.
    gsd_per_layer = [1, .91]

    # Make a df list.
    df_ls = []

    # Process desired plantings.
    for (planting, layer_id, GSD) in zip(plantings, layer_ids, gsd_per_layer):

        # Define path to read in extracted samples.
        input_dir = "/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/" \
                    "{0}-{1}-extracted".format(planting, layer_id)

        # Get extracted sample file names.
        files_in_dir = [i for i in os.listdir(input_dir) if i.endswith(".tif")]

        # Create a list of aom images.
        some_sample_images = []
        for image_name in files_in_dir:
            a_path = os.path.join(input_dir, image_name)
            an_image = cv2.imread(a_path)
            some_sample_images.append((an_image, image_name))

        # Create an out sub-directory.
        directory_path = os.path.join(output_dir, "{0}-yield-estimates-{1}".format(layer_id, planting))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Count pixels.
        params = {
            "sample_images": some_sample_images,
            "thresh_value": 225,
        }

        images_counted_and_marked, pixel_counts, yield_masks = count_white_pix(**params)

        # Generate CSV data.
        df = pd.DataFrame(pixel_counts)
        df.columns = ["pix_counts", "id_tag"]

        # Record GSD.
        df.loc[:, "GSD"] = GSD

        # Calculate 2D yield area.
        df.loc[:, "2D_yield_area"] = df.loc[:, "pix_counts"] * GSD

        # Get per AOM area, manually exported from QGIS at the moment.
        virtual_sample_spaces_in_meters = '/home/will/uas-cotton-photogrammetry/2018_rainMatrix_virtual_aom_size.csv'

        # Get area data for each virtual region of interest.
        df_area = pd.read_csv(virtual_sample_spaces_in_meters)

        # Generate an ID for the join column from the filename written as: spatial_p6_aom_15.tif
        df_area.loc[:, 'id_tag'] = ['id_{0}.tif'.format(str(x).zfill(2)) for x in df_area.loc[:, 'id_tag'].values]

        # Merge data.
        df_both = df.merge(df_area, left_on='id_tag', right_on='id_tag', how='outer')

        # # Yield model y = 0.658 * x - 35.691 based on current findings
        # df_both.loc[:, 'est_yield'] = df_both.loc[:, '2D_yield_area'] * 0.658

        # # Per square meter yield,
        # df_both.loc[:, 'g_per_sq_meter_yield'] = df_both.loc[:, 'est_yield'] / df_both.loc[:, 'area']

        # # Sort values.
        # df_both.sort_values(by=['g_per_sq_meter_yield'], inplace=True)
        #
        # # 1 gram per meter is 8.92179 pounds per acre.
        # df_both.loc[:, 'lb_per_ac_yield'] = df_both.loc[:, 'g_per_sq_meter_yield'] * 8.92179
        #
        # # Lint Yield, turnout.
        # df_both.loc[:, 'turnout_lb_per_ac_yield'] = df_both.loc[:, 'lb_per_ac_yield'] * .38

        # Write pix count data.
        df_both.to_csv(os.path.join(directory_path, "pix-counts-for-{0}.csv".format(layer_id)))

        # Store df from early planting to be combined later.
        df_ls.append(df_both.dropna(axis=0, how='any'))

        # Write marked sample images for inspection.
        for (image, image_name) in images_counted_and_marked:
            cv2.imwrite(os.path.join(directory_path, '{0}-marked.png'.format(image_name)), image)

        # Make directory for pixel location data.
        yield_pixel_location_csv_dir = os.path.join(output_dir, "{0}_white-pixel-locations".format(layer_id))
        if not os.path.exists(yield_pixel_location_csv_dir):
            os.makedirs(yield_pixel_location_csv_dir)

        # Write pixel locations of measured seed-cotton.
        for ((y, x), ID_tag, (h,w)) in yield_masks:

            # Virtual sample space (image) height and width is recorded in df.
            df = pd.DataFrame({'x':x,'y':y,'h':h,'w':w})
            df.to_csv(os.path.join(yield_pixel_location_csv_dir, '{0}_white-pixel-locations.csv'.format(layer_id)))

    df_2018_pixel_counts = pd.concat(df_ls, ignore_index=True)

    # Great numeric re pattern from Stack Overflow for pulling the sample id integers from id tag for join key.
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    # Pull the integers from the id tag but only look at the portion of the tag before the '.tif' part.
    df_2018_pixel_counts.loc[:, 'id'] = df_2018_pixel_counts.loc[:, 'id_tag'].apply(lambda x: rx.findall(x[:-4])[0])
    df_2018_pixel_counts.loc[:, 'id'] = df_2018_pixel_counts.loc[:, 'id'].apply(lambda x: int(x))

    df_2018_pixel_counts.to_csv(os.path.join(output_dir, "2018_pixel_counts_and_2D_yield_area.csv"))

    df_hand_harvested_yield = pd.read_csv("/home/will/uas-cotton-photogrammetry/" \
                                          "2018_rainMatrix_hand_harvested_sample_weights.csv")

    pix_counts_hand_harvested_yield = df_2018_pixel_counts.merge(df_hand_harvested_yield, on='id', how='outer')

    # Sort data frame by id.
    pix_counts_hand_harvested_yield.sort_values(by=['id'], inplace=True)

    # Reset index so plotting is not jumbled.
    pix_counts_hand_harvested_yield.index = list(range(len(pix_counts_hand_harvested_yield)))

    # Save a copy.
    pix_counts_hand_harvested_yield.to_csv(os.path.join(output_dir, "2018_pixel_counts_and_hand_harvested_yield.csv"))


    def clean_poly_eq(coefficients, dec_dig=3):
        n = len(coefficients)
        degs = [i for i in range(n)]
        coefficients = [round(i, dec_dig) for i in coefficients]
        coefficients.reverse()
        pieces = []
        for (cof, deg) in zip(coefficients, degs):
            if deg == 0:
                a = ' + {0}'.format(cof)
                pieces.append(a)
            else:
                a = '{0} x^{1} '.format(cof, deg)
                pieces.append(a)

        equation = 'y = ' + ''.join(pieces[::-1])

        return equation


    def get_poly_hat(x_values, y_values, poly_degree):
        coeffs = np.polyfit(x_values, y_values, poly_degree)
        poly_eqn = np.poly1d(coeffs)

        y_bar = np.sum(y_values) / len(y_values)
        ssreg = np.sum((poly_eqn(x_values) - y_bar) ** 2)
        sstot = np.sum((y_values - y_bar) ** 2)
        r_square = ssreg / sstot

        return (coeffs, poly_eqn, r_square)


    df = pix_counts_hand_harvested_yield

    x = df.loc[:, '2D_yield_area']
    y = df.loc[:, 'seed_cott_weight_(g)']


    coeffs, poly_eqn, r_square = get_poly_hat(x, y, 3)
    line_equation = clean_poly_eq(coeffs, dec_dig=6)

    x_linspace = np.linspace(0, max(x), len(x))
    y_hat = poly_eqn(x_linspace)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(x, y, 'o')
    ax.plot(x_linspace, y_hat, '-')
