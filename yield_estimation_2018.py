#!/home/will/uas-cotton-photogrammetry/cp-venv/bin/python

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
    what = 'aoms'
    year = 2018

    # Define path to output directory.
    output_dir = "/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2018/"

    # Input layers for plantings one, two, three, and four.
    input_layer_20_meters_p1_p3 = "2018-10-23_65_75_20_rainMatrix_odm_orthophoto_modified"  # GSD .75
    input_layer_35_meters_p1_p3 = "2018-10-23_65_75_35_rainMatrix_odm_orthophoto_modified"  # GSD 1.1
    input_layer_50_meters_p1_p3 = "2018-10-23_65_75_50_rainMatrix_odm_orthophoto_modified"  # GSD 1.6
    input_layer_75_meters_p1_p3 = "2018-10-23_65_75_75_rainMatrix_odm_orthophoto_modified"  # GSD 2.5
    input_layer_100_meters_p1_p3 = "2018-10-23_65_75_100_rainMatrix_odm_orthophoto_modified"  # GSD 3.0
    input_layer_30_meters_p1_p3 = "2018-10-26_65_75_30_rainMatrix_odm_orthophoto_modified"  # GSD 1

    # Input layers for plantings five, six, and seven.
    input_layer_35_meters_p4_p7_2018_11_09 = "2018-11-09_65_75_35_rainMatrix_odm_orthophoto_modified"  # 1.1
    input_layer_35_meters_p4_p7_2018_11_15 = "2018-11-15_65_75_35_rainMatrix_modified"  # GSD 1.1

    # The GSD value is given to Open Drone Map in the orthophoto creation phase.
    layer_ids_and_info = [
        (input_layer_20_meters_p1_p3, .75, 20, 200),
        (input_layer_35_meters_p1_p3, 1.0, 35, 200),
        (input_layer_50_meters_p1_p3, 1.6, 50, 200),
        (input_layer_75_meters_p1_p3, 2.5, 75, 190),
        (input_layer_100_meters_p1_p3, 3.0, 100, 180),
        (input_layer_30_meters_p1_p3, 1, 30, 200),
        (input_layer_35_meters_p4_p7_2018_11_09, 1.1, 35, 200),
        (input_layer_35_meters_p4_p7_2018_11_15, 1.0, 35, 240),
    ]

    # The GSD value is given to Open Drone Map in the orthophoto creation phase. The value is stored in a log file.
    # Resulting actual GSD is simply a matter of how things were set during the processing phase, not a reflection of
    # max GSD for a flight simply based on altitude.

    # Make a df list.
    df_ls = []

    # Process desired plantings.
    for (layer_id, GSD, altitude, thresh) in layer_ids_and_info:

        # Define path to read in extracted samples.
        input_dir = "/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/"
        dir_ls = os.listdir(input_dir)
        dir = [i for i in dir_ls if layer_id in i]
        input_dir_paths = [os.path.join(input_dir, i) for i in dir]

        for input_dir_path in input_dir_paths:

            # Get extracted sample file names.
            files_in_dir = [i for i in os.listdir(input_dir_path) if i.endswith(".tif")]

            # Create a list of aom images.
            some_sample_images = []
            for image_name in files_in_dir:
                a_path = os.path.join(input_dir_path, image_name)
                an_image = cv2.imread(a_path)
                some_sample_images.append((an_image, image_name))

            # Create an out sub-directory.
            directory_path = os.path.join(output_dir, "{0}-{1}-{2}-yield-estimates".format(layer_id, GSD, altitude))
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # Count pixels.
            params = {
                "sample_images": some_sample_images,
                "thresh_value": thresh,
            }

            images_counted_and_marked, pixel_counts, yield_masks = count_white_pix(**params)

            # Generate CSV data.
            df = pd.DataFrame(pixel_counts)
            df.columns = ["pix_counts", "id_tag"]

            # Great numeric re pattern from Stack Overflow for pulling the sample id integers from id tag for join key.
            numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
            rx = re.compile(numeric_const_pattern, re.VERBOSE)

            # Pull the integers from the id tag but only look at the portion of the tag before the '.tif' part.
            df.loc[:, 'id'] = df.loc[:, 'id_tag'].apply(lambda x: rx.findall(x[:-4])[0])
            df.loc[:, 'id'] = df.loc[:, 'id'].apply(lambda x: int(x))

            # Record GSD.
            df.loc[:, "GSD"] = GSD

            # Record altitude.
            df.loc[:, "altitude"] = altitude

            # Layer id.
            df.loc[:, "layer_id"] = layer_id

            # Calculate 2D yield area.
            df.loc[:, "2D_yield_area"] = df.loc[:, "pix_counts"] * GSD

            # Get per AOM area, manually exported from QGIS at the moment.
            virtual_sample_spaces_in_meters = '/home/will/uas-cotton-photogrammetry/2018_rainMatrix_virtual_aom_size.csv'

            # Get area data for each virtual region of interest.
            df_area = pd.read_csv(virtual_sample_spaces_in_meters)

            # Merge data.
            df_both = df.merge(df_area, left_on='id', right_on='id', how='outer')

            # Insert year.
            df_both.loc[:, 'year'] = year

            # Write pix count data.
            df_both.to_csv(os.path.join(directory_path, "pix-counts-for-{0}.csv".format(layer_id)))

            # Store df from early planting to be combined later.
            df_ls.append(df_both.dropna(axis=0, how='any'))

            # Write marked sample images for inspection.
            for (image, image_name) in images_counted_and_marked:
                cv2.imwrite(os.path.join(directory_path, '{0}-marked.png'.format(image_name)), image)

            # Write pixel locations of measured seed-cotton.
            for ((y, x), ID_tag, (h,w)) in yield_masks:

                # Virtual sample space (image) height and width is recorded in df.
                df = pd.DataFrame({'x':x,'y':y,'h':h,'w':w})
                df.to_csv(os.path.join(output_dir, '{0}_white-pixel-locations.csv'.format(layer_id)))

    df_2018_pixel_counts = pd.concat(df_ls, ignore_index=True)

    df_2018_pixel_counts.to_csv(os.path.join(output_dir, "2018_pixel_counts_and_2D_yield_area.csv"))

    df_hand_harvested_yield = pd.read_csv("/home/will/uas-cotton-photogrammetry/" \
                                          "2018_rainMatrix_hand_harvested_sample_weights.csv")

    pix_counts_hand_harvested_yield = df_2018_pixel_counts.merge(df_hand_harvested_yield, on='id', how='outer')

    # Sort data frame by id.
    pix_counts_hand_harvested_yield.sort_values(by=['id'], inplace=True)

    # Reset index so plotting is not jumbled.
    pix_counts_hand_harvested_yield.index = list(range(len(pix_counts_hand_harvested_yield)))

    # Save a copy.
    pix_counts_hand_harvested_yield.to_csv(os.path.join(output_dir, "2018_pixel_counts_and_hand_harvested_yield.csv"), index=False)


