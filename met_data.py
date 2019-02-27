import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

df17 = pd.read_csv('/home/will/uas-cotton-photogrammetry/met_data_2017.csv', header=None)
df18 = pd.read_csv('/home/will/uas-cotton-photogrammetry/met_data_2018.csv', header=None)

column_names_17 = []
for (i,j,k,l) in zip(df17.loc[0,:], df17.loc[1,:], df17.loc[2,:], df17.loc[3,:]):
    name = [b for b in [i,j,k,l] if isinstance(b, str)]
    name = "-".join(name)
    column_names_17.append(name)

column_names_18 = []
for (i,j,k,l) in zip(df18.loc[0,:], df18.loc[1,:], df18.loc[2,:], df18.loc[3,:]):
    name = [b for b in [i,j,k,l] if isinstance(b, str)]
    name = "-".join(name)
    column_names_18.append(name)

df17 = df17.loc[5:, :]
df18 = df18.loc[5:, :]

df17.columns = column_names_17
df18.columns = column_names_18

df_both = pd.concat([df17, df18])

# Plant DOY, harvest DOY 2017.
plant_harv_17 = [
    (1, 95, 238),
    (2, 109, 241),
    (3, 122, 262),
    (4, 136, 282),
    (5, 151, 296),
    (6, 173, 301),
    (7, 192, 301)
]

# Plant DOY, harvest DOY 2018.
plant_harv_18 = [
    (1, 70, 214),
    (2, 92, 221),
    (3, 102, 231),
    (4, 121, 237),
    (5, 134, 268),
    (6, 149, 382),
    (7, 164, 296)
]

# Clean.
df_clean = pd.DataFrame()

df_clean.loc[:, 'Year'] = df_both.loc[:, 'Year'].apply(lambda x: int(x))
df_clean.loc[:, 'Day'] = df_both.loc[:, 'Day'].apply(lambda x: int(x))
df_clean.loc[:, 'Rain-Standard'] = df_both.loc[:, 'Rain-Standard'].apply(lambda x: float(x))
df_clean.loc[:, 'Min-Air-Temp-@ 2 m'] = df_both.loc[:, 'Min-Air-Temp-@ 2 m'].apply(lambda x: float(x))
df_clean.loc[:, 'Max-Air-Temp-@ 2 m'] = df_both.loc[:, 'Max-Air-Temp-@ 2 m'].apply(lambda x: float(x))
df_clean.loc[:, 'Avg-Air-Temp-@ 2 m'] = df_both.loc[:, 'Avg-Air-Temp-@ 2 m'].apply(lambda x: float(x))
df_clean.loc[:, 'Station-Press'] = df_both.loc[:, 'Station-Press'].apply(lambda x: float(x))
df_clean.loc[:, 'Min-RH-@ 2 m'] = df_both.loc[:, 'Min-RH-@ 2 m'].apply(lambda x: float(x))
df_clean.loc[:, 'Max-RH-@ 2 m'] = df_both.loc[:, 'Max-RH-@ 2 m'].apply(lambda x: float(x))

# Subsets for each planting.
plant_subsets_17 = []
for (planting, plant, harvest) in plant_harv_17:
    df = df_clean.loc[
        (df_clean['Day'] >= plant) &
        (df_clean['Day'] <= harvest) &
        (df_clean['Year'] == 2017), :]
    df.loc[:, 'planting'] = planting
    plant_subsets_17.append(df)

# Subsets for each planting.
plant_subsets_18 = []
for (planting, plant, harvest) in plant_harv_18:
    df = df_clean.loc[
        (df_clean['Day'] >= plant) &
        (df_clean['Day'] <= harvest) &
        (df_clean['Year'] == 2018), :]
    df.loc[:, 'planting'] = planting
    plant_subsets_18.append(df)

# Calculate GDU for each planting.
gdu_base = 15.6
for df in plant_subsets_17:
    df.loc[:, 'GDU'] = ((df.loc[:, 'Min-Air-Temp-@ 2 m'] + df.loc[:, 'Max-Air-Temp-@ 2 m'])/2) - 15.6
    df.loc[:, 'GDU'] = df.loc[:, 'GDU'].apply(lambda x: x if x > 0 else 0)

for df in plant_subsets_18:
    df.loc[:, 'GDU'] = ((df.loc[:, 'Min-Air-Temp-@ 2 m'] + df.loc[:, 'Max-Air-Temp-@ 2 m'])/2) - 15.6
    df.loc[:, 'GDU'] = df.loc[:, 'GDU'].apply(lambda x: x if x > 0 else 0)

# Get cumulative data.
cumulative17 = []
for df in plant_subsets_17:
    year = df.loc[:, 'Year'].values[0]
    planting = df.loc[:, 'planting'].values[0]
    acum_gdu = df.loc[:, 'GDU'].sum()
    acum_rain = df.loc[:, 'Rain-Standard'].sum()

    info = (year, planting, acum_gdu, acum_rain)

    cumulative17.append(info)

# Get cumulative data.
cumulative18 = []
for df in plant_subsets_18:
    year = df.loc[:, 'Year'].values[0]
    planting = df.loc[:, 'planting'].values[0]
    acum_gdu = df.loc[:, 'GDU'].sum()
    acum_rain = df.loc[:, 'Rain-Standard'].sum()

    info = (year, planting, acum_gdu, acum_rain)

    cumulative18.append(info)

info17 = pd.DataFrame(cumulative17)
info18 = pd.DataFrame(cumulative18)

info17.columns = ['Year', 'Planting', 'GDU', 'actual_rainfall']
info18.columns = ['Year', 'Planting', 'GDU', 'actual_rainfall']

info_all = pd.concat([info17, info18])
info_all.to_csv('/home/will/uas-cotton-photogrammetry/cumulative_info_by_planting_all.csv', index=False)



