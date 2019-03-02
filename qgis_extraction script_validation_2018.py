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

    # Input layer for plantings one, two, three, and four.
    input_layer_35_meters_earlier = "2018-10-23_65_75_35_rainMatrix_odm_orthophoto_modified"  # GSD 1.1

    # Input layer for plantings five, and six.
    input_layer_35_meters_p4_p6 = "2018-11-09_65_75_35_rainMatrix_odm_orthophoto_modified"  # 1.1

    # Input layer for planting seven.
    input_layer_35_meters_later = "2018-11-15_65_75_35_rainMatrix_modified"  # GSD 1.1

    # Define path to output directory.
    an_output_dir = "/home/will/uas-cotton-photogrammetry/output/extracted_validation_samples_2018/"

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
    my_group = my_layer_tree.findGroup("2018_aoms_by_decile_yield_model_validation")

    # Divide AOMS for early plantings and late plantings.
    planting_7 = my_group.children()[0:4]
    planting_6 = my_group.children()[4:8]
    planting_5 = my_group.children()[8:11]
    planting_4 = my_group.children()[11:14]
    planting_3 = my_group.children()[14:18]
    planting_2 = my_group.children()[18:22]
    planting_1 = my_group.children()[22:26]

    # Drop the eighth late planting.
    groups = [
        (planting_1, input_layer_35_meters_earlier, 'p1'),
        (planting_2, input_layer_35_meters_earlier, 'p2'),
        (planting_3, input_layer_35_meters_earlier, 'p3'),
        (planting_4, input_layer_35_meters_later, 'p4'),
        (planting_5, input_layer_35_meters_later, 'p4'),
        (planting_6, input_layer_35_meters_later, 'p5'),
        (planting_7, input_layer_35_meters_later, 'p6'),
    ]

    # Process for desired plantings
    for (chunk, layer_id, planting) in groups:

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
                            """.format('__'.join(layer_id), len(groups), formatted_time))

    # Close project.
    qgs.exitQgis()
