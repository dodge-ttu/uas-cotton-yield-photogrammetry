import os
import sys
from datetime import datetime
from qgis.core import QgsProject
from qgis.core import QgsApplication
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
    plantings = ['earlier', 'later']
    what = 'aoms'

    # Append QGIS to path.
    sys.path.append("/home/will/uas-cotton-photogrammetry/cp-venv/lib/python3/dist-packages")

    # Define input layer.
    # input_layer = "2018-11-15_65_75_35_rainMatrix_modified"  # yield
    input_layer_name_earlier = "2018-10-26_65_75_30_rainMatrix_odm_orthophoto_modified"
    input_layer_name_later = "2018-11-15_65_75_35_rainMatrix_modified"

    layer_ids = [input_layer_name_earlier, input_layer_name_later]

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

    # Return the layer tree and isolate the group of interest to programmatically extract the individual.
    my_layer_tree = QgsProject.instance().layerTreeRoot()
    my_group = my_layer_tree.findGroup("individual_sample_spaces_2018")

    # Generate a list of items in the group of interest.
    a_layer_list = my_group.children()

    layer_chunks = [a_layer_list[:48], a_layer_list[47:]]

    # Process for desired plantings
    for (planting, layer_id, layer_chunk) in zip(plantings, layer_ids, layer_chunks):

        # Get date to tag output.
        raw_time = datetime.now()
        formatted_time = datetime.strftime(raw_time, "%Y-%m-%d %H:%M:%S")

        # Create an out sub-directory.
        directory_path = os.path.join(an_output_dir, "{0}-{1}-extracted".format(planting, layer_id))
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        print(directory_path)

        # Process sample spaces for early plantings.
        params = {'output_dir': directory_path,
                  'layer_list': layer_chunk,
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
