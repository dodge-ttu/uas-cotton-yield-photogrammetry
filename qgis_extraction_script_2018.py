#!/home/will/uas-cotton-photogrammetry/cp-venv/bin/python

import os
import sys
from datetime import datetime
from qgis.core import QgsProject
from qgis.core import QgsApplication

# Append processing module location to path.
sys.path.append("/home/will/uas-cotton-photogrammetry/cp-venv/lib/python3/dist-packages")

import processing
from processing.core.Processing import Processing

# # Display all available processing algorithms.
# def disp_algs():
#     for alg in QgsApplication.processingRegistry().algorithms():
#         print("{}:{} --> {}".format(alg.provider().name(), alg.name(), alg.displayName()))
#
# disp_algs()

# Function to iteratively extract virtual sample spaces.
def make_samples(layer_list=None, output_dir=None, input_layer_name=None):
    for i in layer_list:
        parameters = {
            'ALPHA_BAND': False,
            'CROP_TO_CUTLINE': True,
            'DATA_TYPE': 0,
            'INPUT': '{0}'.format(input_layer_name),
            'KEEP_RESOLUTION': True,
            'MASK': '{0}'.format(i.name()),
            'NODATA': None,
            'OPTIONS': '',
            'OUTPUT': os.path.join(output_dir, "{0}.tif".format(i.name())),
        }

        processing.run('gdal:cliprasterbymasklayer', parameters)


if __name__ == '__main__':

    # Details.
    what = 'aoms'

    # Input layers for plantings one, two, three, and four.
    input_layer_20_meters_p1_p3 = "2018-10-23_65_75_20_rainMatrix_odm_orthophoto_modified"  # GSD .75
    input_layer_35_meters_p1_p3 = "2018-10-23_65_75_35_rainMatrix_odm_orthophoto_modified"  # GSD 1.1
    input_layer_50_meters_p1_p3 = "2018-10-23_65_75_50_rainMatrix_odm_orthophoto_modified"  # GSD 1.6
    input_layer_75_meters_p1_p3 = "2018-10-23_65_75_75_rainMatrix_odm_orthophoto_modified"  # GSD 2.5
    input_layer_100_meters_p1_p3 = "2018-10-23_65_75_100_rainMatrix_odm_orthophoto_modified"  # GSD 3.0

    # Input layers for plantings five, six, and seven.
    input_layer_35_meters_p4_p7 = "2018-11-15_65_75_35_rainMatrix_modified"  # GSD 1.1

    layer_ids_plantings_p1_p4 = [
        input_layer_20_meters_p1_p3,
        input_layer_35_meters_p1_p3,
        input_layer_50_meters_p1_p3,
        input_layer_75_meters_p1_p3,
        input_layer_100_meters_p1_p3,

    ]

    layer_ids_plantings_p4_p7 = [
        input_layer_35_meters_p4_p7,
    ]

    # Define path to output directory.
    an_output_dir = "/home/will/uas-cotton-photogrammetry/output/extracted_samples_2018/"

    # Get date to tag output.
    raw_time = datetime.now()
    formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

    # Create a reference to the QGIS application.
    qgs = QgsApplication([], False)

    # Load providers.
    qgs.initQgis()

    # Create a project instance.
    project = QgsProject.instance()

    # Load a project.
    project.read('/home/will/MAHAN MAP 2018/MAHAN MAP 2018.qgs')

    # Initialize processing.
    Processing.initialize()

    # Return the layer tree and isolate the group of interest to programmatically extract the individual samples.
    my_layer_tree = QgsProject.instance().layerTreeRoot()
    my_group = my_layer_tree.findGroup("individual_sample_spaces_2018")

    # Generate a list of items in the group of interest.
    a_layer_list = my_group.children()

    # Break list out by planting date.
    planting_1 = a_layer_list[:30]
    planting_2 = a_layer_list[30:39]
    planting_3 = a_layer_list[39:47]
    planting_4 = a_layer_list[47:55]
    planting_5 = a_layer_list[55:61]
    planting_6 = a_layer_list[61:69]
    planting_7 = a_layer_list[69:76]

    # List of list tuples containing chunk and respective planting id.
    layer_ls_chunks = [
        (planting_1, 'p1'),
        (planting_2, 'p2'),
        (planting_3, 'p3'),
        (planting_4, 'p4'),
        (planting_5, 'p5'),
        (planting_6, 'p6'),
        (planting_7, 'p7'),
    ]

    # Process for desired plantings
    for (chunk, planting) in layer_ls_chunks:

        if planting in ('p1', 'p2', 'p3'):
            layer_ids = layer_ids_plantings_p1_p4
        else:
            layer_ids = layer_ids_plantings_p4_p7

        for layer_id in layer_ids:

            # Get date to tag output.
            raw_time = datetime.now()
            formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

            # Create an out sub-directory.
            directory_path = os.path.join(an_output_dir, "{0}-{1}-extracted".format(layer_id, planting))
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            print(directory_path)

            params = {'output_dir': directory_path,
                      'layer_list': chunk,
                      'input_layer_name': layer_id}

            make_samples(**params)

            # Write a meta-data file with the details of this extraction for future reference.
            with open(os.path.join(directory_path, "sample_meta_data.txt"), "w") as tester:
                tester.write("""Sample Layer ID: {0}\n
                                Number of Samples: {1}\n
                                Samples Generated On: {2}\n
                                """.format('__'.join(layer_ids), len(a_layer_list), formatted_time))

    # Close project.
    qgs.exitQgis()
