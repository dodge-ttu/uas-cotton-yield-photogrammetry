import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_ = pd.read_csv('/home/will/uas-cotton-photogrammetry/distances_non_GCP_measure_1_2018-07-25_75_75_50_pivot_odm_orthophoto_modified.csv', header=[0])
df_2 = pd.read_csv('/home/will/uas-cotton-photogrammetry/distances_non_GCP_measure_2_2018-08-13_75_75_50_pivot_odm_orthophoto_modified.csv', header=[0])
df_3 = pd.read_csv('/home/will/uas-cotton-photogrammetry/distances_non_GCP_measure_3_2018-09-04_75_75_50_rainMatrix_odm_orthophoto_modified.csv', header=[0])