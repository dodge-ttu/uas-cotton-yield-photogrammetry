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
    output_dir = "/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2017/"

    # Details.
    plantings = ['p1']*12 + ['p2']*17
    what = 'aoms'
    altitude_per_flight = [20,22,24,26,28,30]

    # Define input layer.
    input_layer_20_meters = "2017-11-17_75_75_20_odm_orthophoto_modified" # 153.846pixels/meter
    input_layer_22_meters = "2017-11-17_75_75_22_odm_orthophoto_modified" # 133.333pixels/meter
    input_layer_24_meters = "2017-11-16_75_75_24_odm_orthophoto_modified" # 117.647pixels/meter
    input_layer_26_meters = "2017-11-16_75_75_26_odm_orthophoto_modified" # 105.263pixels/meter
    input_layer_28_meters = "2017-11-16_75_75_28_odm_orthophoto_modified" # 100pixels/meter
    input_layer_30_meters = "2017-11-16_75_75_30_odm_orthophoto_modified" # 98pixels/meter

    layer_ids = [
        input_layer_20_meters,
        input_layer_22_meters,
        input_layer_24_meters,
        input_layer_26_meters,
        input_layer_28_meters,
        input_layer_30_meters,
    ]

    # The GSD value is given to Open Drone Map in the orthophoto creation phase. The value is stored in a log file.
    # Resulting actual GSD is simply a matter of how things were set during the processing phase, not a reflection of
    # max GSD for a flight simply based on altitude.

    # GSD (cm/pixel) values for each input layer.
    gsd_per_layer = [.65, .75, .85, .95, 1, 1]

    # Make a df list.
    df_ls = []

    # Process desired plantings.
    for (layer_id, GSD, altitude) in zip(layer_ids, gsd_per_layer, altitude_per_flight):

        # Define path to read in extracted samples.
        input_dir = "/home/will/uas-cotton-photogrammetry/output/extracted_samples_2017/" \
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

        # Record altutude.
        df.loc[:, "altitude"] = altitude

        # Calculate 2D yield area.
        df.loc[:, "2D_yield_area"] = df.loc[:, "pix_counts"] * GSD

        # All sample spaces were the same in 2017 at one meter.

        # # Get area data for each virtual region of interest.
        # df_area = pd.read_csv(virtual_sample_spaces_in_meters)
        #
        # # Generate an ID for the join column from the filename written as: spatial_p6_aom_15.tif
        # df_area.loc[:, 'id_tag'] = ['id_{0}.tif'.format(str(x).zfill(2)) for x in df_area.loc[:, 'id_tag'].values]
        #
        # # Merge data.
        # df_both = df.merge(df_area, left_on='id_tag', right_on='id_tag', how='outer')

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

        # # Write pix count data.
        # df_both.to_csv(os.path.join(directory_path, "pix-counts-for-{0}.csv".format(layer_id)))

        # Store df from early planting to be combined later.
        df_ls.append(df.dropna(axis=0, how='any'))

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

    df_2017_pixel_counts = pd.concat(df_ls, ignore_index=True)

    # Great numeric re pattern from Stack Overflow for pulling the sample id integers from id tag for join key.
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    # Pull the integers from the id tag but only look at the portion of the tag before the '.tif' part.
    df_2017_pixel_counts.loc[:, 'id'] = df_2017_pixel_counts.loc[:, 'id_tag'].apply(lambda x: rx.findall(x[:-4])[0])
    df_2017_pixel_counts.loc[:, 'id'] = df_2017_pixel_counts.loc[:, 'id'].apply(lambda x: int(x))

    # Unique id.
    df_2017_pixel_counts.loc[:, "unique_id"] = df_2017_pixel_counts["id"].map(str) + "_" + df_2017_pixel_counts["altitude"].map(str)

    df_2017_pixel_counts.to_csv(os.path.join(output_dir, "2018_pixel_counts_and_2D_yield_area.csv"))

    df_hand_harvested_yield = pd.read_csv("/home/will/uas-cotton-photogrammetry/" \
                                          "2017_rainMatrix_hand_harvested_sample_weights.csv")

    pix_counts_hand_harvested_yield = df_2017_pixel_counts.merge(df_hand_harvested_yield, on='id', how='outer')

    # Sort data frame by id.
    pix_counts_hand_harvested_yield.sort_values(by=['id'], inplace=True)

    # Reset index so plotting is not jumbled.
    pix_counts_hand_harvested_yield.index = list(range(len(pix_counts_hand_harvested_yield)))

    # Save a copy.
    pix_counts_hand_harvested_yield.to_csv(os.path.join(output_dir, "2017_pixel_counts_and_hand_harvested_yield.csv"))


