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
    output_dir = "/home/will/uas-cotton-photogrammetry/output/yield_validation_counted_marked_2018/"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Input layer for plantings one, two, three, and four.
    input_layer_35_meters_p1_p3 = "2018-10-23_65_75_35_rainMatrix_odm_orthophoto_modified"  # GSD 1.1

    # Input layer for plantings five, and six.
    # input_layer_35_meters_p4_p6 = "2018-11-09_65_75_35_rainMatrix_odm_orthophoto_modified"  # 1.1

    # Input layer for planting seven.
    input_layer_35_meters_p7 = "2018-11-15_65_75_35_rainMatrix_modified"  # GSD 1.1

    # The GSD value is given to Open Drone Map in the orthophoto creation phase.
    layer_ids_and_info = [
        (input_layer_35_meters_p1_p3, 1.0, 35, 205),
        # (input_layer_35_meters_p4_p6, 1.0, 35, 220),
        (input_layer_35_meters_p7, 1.0, 35, 225),
    ]

    # Make a df list.
    df_ls = []

    # Process desired plantings.
    for (layer_id, GSD, altitude, thresh) in layer_ids_and_info:

        # Define path to read in extracted samples.
        input_dir = "/home/will/uas-cotton-photogrammetry/output/extracted_validation_samples_2018/"
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

            # Planting from filename.
            df.loc[:, 'planting'] = df.loc[:, 'id_tag'].apply(lambda x: x.split('_')[0])

            # Decile from filename.
            df.loc[:, 'decile'] = df.loc[:, 'id_tag'].apply(lambda x: x[3:5])

            # Make ID column.
            df.loc[:, 'id'] = df.loc[:, 'planting'].map(str) + '_' + df.loc[:, 'decile'].map(str)

            # Record GSD.
            df.loc[:, "GSD"] = GSD

            # Record altitude.
            df.loc[:, "altitude"] = altitude

            # Layer id.
            df.loc[:, "layer_id"] = layer_id

            # Get per AOM area, manually exported from QGIS at the moment.
            virtual_sample_spaces_in_meters = '/home/will/uas-cotton-photogrammetry/2018_rainMatrix_virtual_aom_size_VALIDATION.csv'

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

    df_validation_2018_pixel_counts = pd.concat(df_ls, ignore_index=True)

    df_validation_2018_pixel_counts.to_csv(os.path.join(output_dir, "2018_pixel_counts_and_2D_yield_area.csv"))

    # Machine harvested by planting for model testing 2018.
    df_validation = pd.read_csv("/home/will/uas-cotton-photogrammetry/2018_rain_matrix_timeline_data_by_decile_machine_harvest_VALIDATION.csv")

    # pcvy = pix_counts_validation_yield
    pcvy = df_validation_2018_pixel_counts.merge(df_validation, on='id', how='outer')

    # Sort data frame by id.
    pcvy.sort_values(by=['id'], inplace=True)

    # Reset index so plotting is not jumbled.
    pcvy.index = list(range(len(pcvy)))

    # Implement yield model as calulated for a 35 meter flight.
    altitude = 35
    slope = 5.885 - (.1348 * altitude) + (0.0009007 * (altitude**2))

    pcvy.loc[:, 'PCCA'] = pcvy.loc[:, 'pix_counts'] * pcvy.loc[:, 'GSD']

    pcvy.loc[:, 'predicted_seed_cott_grams'] = pcvy.loc[:, 'PCCA'] * slope
    pcvy.loc[:, 'predicted_seed_cott_grams_per_meter'] = pcvy.loc[:, 'predicted_seed_cott_grams'] / pcvy.loc[:, 'area']

    # Convert grams per meter to kilograms per hectare by mutliplying by ten.
    pcvy.loc[:, 'predicted_kilo_per_ha'] = pcvy.loc[:, 'predicted_seed_cott_grams_per_meter']

    # Convert pounds to kg.
    pcvy.loc[:, 'machine_harv_kilo'] = pcvy.loc[:, 'yield_pounds'] * 0.453592

    # Convert m^2 to ha.
    pcvy.loc[:, 'area_hectares'] = pcvy.loc[:, 'area'] * 0.0001

    # Covert yield to kg / ha.
    pcvy.loc[:, 'yield_kg_per_ha'] = pcvy.loc[:, 'machine_harv_kilo'] / pcvy.loc[:, 'area_hectares']

    # Save a copy.
    pcvy.to_csv(os.path.join(output_dir, "2018_valedation_pixel_counts_and_machine_harvested_yield.csv"), index=False)

    # Quick plots.
    x = pcvy.loc[:, 'yield_kg_per_ha']
    y = pcvy.loc[:, 'predicted_kilo_per_ha']

    plt.plot(x, y, 'o')