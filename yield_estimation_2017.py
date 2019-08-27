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
        
        (T, thresh) = cv2.threshold(b, thresh_value, 255, cv2.THRESH_BINARY)

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
    year = 2017

    # Define path to output directory.
    output_dir = "/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2017/"

    # Input layers for plantings one and two.
    input_layer_20_meters_p1_p2 = "2017-11-17_75_75_20_odm_orthophoto_modified" # GSD .65
    input_layer_22_meters_p1_p2 = "2017-11-17_75_75_22_odm_orthophoto_modified" # GSD .75
    input_layer_24_meters_p1_p2 = "2017-11-16_75_75_24_odm_orthophoto_modified" # GSD .85
    input_layer_26_meters_p1_p2 = "2017-11-16_75_75_26_odm_orthophoto_modified" # GSD .95
    input_layer_28_meters_p1_p2 = "2017-11-16_75_75_28_odm_orthophoto_modified" # GSD 1.0
    input_layer_30_meters_p1_p2 = "2017-11-16_75_75_30_odm_orthophoto_modified" # GSD 1.0

    # Input layers for plantings three and four.
    input_layer_20_meters_p3_p4 = "2017-12-01_75_75_20_validation_odm_orthophoto_modified" # GSD .65
    input_layer_24_meters_p3_p4 = "2017-12-01_75_75_24_validation_odm_orthophoto_modified" # GSD .75

    # The GSD value is given to Open Drone Map in the orthophoto creation phase. The value is stored in a log file.
    # Resulting actual GSD is simply a matter of how things were set during the processing phase, not a reflection of
    # max GSD for a flight simply based on altitude.
    layer_ids_and_info = [
        (input_layer_20_meters_p1_p2, .65, 20, 230),
        (input_layer_22_meters_p1_p2, .70, 22, 230),
        (input_layer_24_meters_p1_p2, .75, 24, 230),
        (input_layer_26_meters_p1_p2, .80, 26, 230),
        (input_layer_28_meters_p1_p2, .85, 28, 230),
        (input_layer_30_meters_p1_p2, .90, 30, 230),
        (input_layer_20_meters_p3_p4, .65, 20, 250),
        (input_layer_24_meters_p3_p4, .85, 24, 250),
    ]

    # Make a df list.
    df_ls = []

    # Process desired plantings.
    for (layer_id, GSD, altitude, thresh) in layer_ids_and_info:

        # Define path to read in extracted samples.
        input_dir = "/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/"
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

            # Record GSD.
            df.loc[:, "GSD"] = GSD

            # Record altitude.
            df.loc[:, "altitude"] = altitude

            # Layer id.
            df.loc[:, "layer_id"] = layer_id

            # Calculate 2D yield area.
            df.loc[:, "2D_yield_area"] = df.loc[:, "pix_counts"] * GSD

            # All sample spaces were the same in 2017 at one meter.

            # Insert year.
            df.loc[:, 'year'] = 2017

            df_ls.append(df.dropna(axis=0, how='any'))

            # Write marked sample images for inspection.
            for (image, image_name) in images_counted_and_marked:
                cv2.imwrite(os.path.join(directory_path, '{0}-marked.png'.format(image_name)), image)

            # Write pixel locations of measured seed-cotton.
            for ((y, x), ID_tag, (h,w)) in yield_masks:

                # Virtual sample space (image) height and width is recorded in df.
                df_coords = pd.DataFrame({'x':x,'y':y,'h':h,'w':w})
                df_coords.to_csv(os.path.join(output_dir, '{0}_white-pixel-locations.csv'.format(layer_id)))

    df_2017_pixel_counts = pd.concat(df_ls, ignore_index=True)

    # Great numeric re pattern from Stack Overflow for pulling the sample id integers from id tag for join key.
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    # Pull the integers from the id tag but only look at the portion of the tag before the '.tif' part.
    df_2017_pixel_counts.loc[:, 'id'] = df_2017_pixel_counts.loc[:, 'id_tag'].apply(lambda x: rx.findall(x[:-4])[0])
    df_2017_pixel_counts.loc[:, 'id'] = df_2017_pixel_counts.loc[:, 'id'].apply(lambda x: int(x))

    df_2017_pixel_counts.to_csv(os.path.join(output_dir, "2017_pixel_counts_and_2D_yield_area.csv"))

    df_hand_harvested_yield = pd.read_csv("/home/will/uas-cotton-photogrammetry/" \
                                          "2017_rainMatrix_hand_harvested_sample_weights.csv")

    pix_counts_hand_harvested_yield = df_2017_pixel_counts.merge(df_hand_harvested_yield, on='id', how='outer')

    # Sort data frame by id.
    pix_counts_hand_harvested_yield.sort_values(by=['id'], inplace=True)

    # Reset index so plotting is not jumbled.
    pix_counts_hand_harvested_yield.index = list(range(len(pix_counts_hand_harvested_yield)))

    # Save a copy.
    pix_counts_hand_harvested_yield.to_csv(os.path.join(output_dir, "2017_pixel_counts_and_hand_harvested_yield.csv"), index=False)


