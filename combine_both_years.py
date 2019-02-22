import os
import pandas as pd

out_dir = '/home/will/uas-cotton-photogrammetry/output'

df17 = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2017/2017_pixel_counts_and_hand_harvested_yield.csv')
df18 = pd.read_csv('/home/will/uas-cotton-photogrammetry/output/yield_counted_marked_2018/2018_pixel_counts_and_hand_harvested_yield.csv')

df_all = pd.concat([df17, df18], sort=False)

df_all.loc[:, 'planting_number'] = df_all.loc[:, 'planting'].apply(lambda x: int(x[1]))

df_all.loc[:, 'year_planting'] = df_all.loc[:, 'year'].map(str) + df_all.loc[:, 'planting_number'].map(str)

df_info = pd.read_csv('/home/will/uas-cotton-photogrammetry/cumulative_info_by_planting_all.csv', header=[0])

df_info.loc[:, 'year_planting'] = df_info.loc[:, 'Year'].map(str) + df_info.loc[:, 'Planting'].map(str)

df_all = df_all.merge(df_info, left_on='year_planting', right_on='year_planting')

order_vectors = [
    'id_tag',
    'year',
    'planting_number',
    'pix_counts',
    'altitude',
    'GSD',
    '2D_yield_area',
    'seed_cott_weight_(g)',
    'decile',
    'variety',
    'GDU',
    'actual_rainfall',
    'layer_id',
    'marker_id',
    #'sample_length_(m)',
    #'planting',
    #'area',
    #'id',
]

df_all = df_all.loc[:, order_vectors]

df_all.to_csv(os.path.join(out_dir, 'uas_yield_and_hand_harvested_all.csv'), index=False)
