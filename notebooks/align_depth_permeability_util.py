import pandas as pd
import os
import lasio
import numpy as np
import glob

def find_nearest(array, value):
    """
    Finds the nearest value in an array to a given value and aligns
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]

def match_facies(depth, df):
    facies = df[(df['Start Depth'] <= depth) & (df['End Depth'] >= depth)]['Facies Class Name']
    return facies.values[0] if len(facies) > 0 else None

def align_depths(data_dir, well_name):
    """
    Aligns wirelineLog depth to permeability depth
    """
    try:
        wirelineLog_df = lasio.read(f'{data_dir}wirelineLog/{well_name}_logs.las').df()
    except:
        print(f'No wireline log for {well_name}')
        return None
    
    wirelineLog_df.reset_index(drop=False, inplace=True)
    wirelineLog_df['DEPTH'] = pd.to_numeric(wirelineLog_df['DEPTH'], errors='coerce')
    wirelineLog_df.dropna(subset=['DEPTH'], inplace=True)
    
    try:
        perm_df = pd.read_csv(f'{data_dir}labels/permeability/{well_name}.csv'.replace('-', '_'))
    except:
        print(f'No permeability label for {well_name}')
        return None
    
    perm_df.columns = perm_df.columns.str.replace('\n', '_')
    perm_df['DEPTH'] = pd.to_numeric(perm_df['DEPTH'], errors='coerce')
    perm_df.dropna(subset=['DEPTH'], inplace=True)
    perm_df.dropna(axis=1, how='all', inplace=True)
    perm_df.columns = perm_df.columns.to_series().apply(lambda x: x.lstrip())

    perm_df['nearest'] = perm_df['DEPTH'].apply(lambda x: find_nearest(wirelineLog_df['DEPTH'], x))

    merge_df = pd.merge(perm_df, wirelineLog_df, left_on='nearest', right_on='DEPTH', how='left')
    merge_df['depth_difference'] = merge_df['DEPTH_y'] - merge_df['DEPTH_x']
    merge_df.drop(['DEPTH_y'], axis=1, inplace=True)
    merge_df.rename(columns={'DEPTH_x': 'DEPTH'}, inplace=True)
    merge_df.set_index('DEPTH', inplace=True)
    return merge_df    

def align_facies(data_dir, well_name, df_merge):
    try:
        facies_df = pd.read_csv(f'{data_dir}labels/faciesclass/{well_name}.csv')
    except:
        print(f'No facies label for {well_name}')
        return None
    
    df_merge.reset_index(inplace=True)

    df_merge['Facies Class Name'] = df_merge['DEPTH'].apply(lambda x: match_facies(x, facies_df))

    return df_merge.loc[df_merge['Facies Class Name']!='nc']

def well_name_list(data_folder):
    """
    Returns 
    (1) a list of well locations
    (2) a list of well names 
    """
    # labels_pathlist = glob.glob(f'{data_folder}/labels/*/*.csv')
    # wirelineLog_pathlist = glob.glob(f'{data_folder}/wirelineLog/*.las')
    print(os.getcwd())
    wells_locations = f'{data_folder}well_loc.csv'

    # well_name_list = [path.split('/')[-1].split('_')[0] for path in wirelineLog_pathlist]
    loc_df = pd.read_csv(wells_locations)
    well_name_list = loc_df.iloc[:,0].values.tolist()
    return well_name_list, loc_df

def concat_data(data_dir, well_list):
    depth_dir_list = glob.glob(f'{data_dir}well_depth/*.npy')

    _, locations = well_name_list(data_dir)

    dataframes = []

    for well_name in well_list:

        well_depth = np.concatenate([np.load(f) for f in depth_dir_list if well_name in f])
        well_min, well_max = well_depth.min(), well_depth.max()

        df = align_depths(data_dir, well_name)

        if df is None:
            continue
        
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        df = align_facies(data_dir, well_name, df)

        df = df.loc[(df['DEPTH'] >= well_min) & (df['DEPTH'] <= well_max)]
        df = df.loc[~df['Facies Class Name'].isin(['nc', 'bc'])]

        if df is None:
            continue

        if 'DTS' not in df.columns:
            if 'DTS1' in df.columns:
                df.rename(columns={'DTS1':'DTS'}, inplace=True)
            elif 'DTS_1' in df.columns:
                df.rename(columns={'DTS_1':'DTS'}, inplace=True)
        
        # df['VpVsRatio'] = df['DTC'] / df['DTS']

        if ('RMIC' in df.columns) and ('RMED' not in df.columns):
            df.rename(columns={'RMIC':'RMED'}, inplace=True)

        df.reset_index(inplace=True)

        df['lon'] = locations.loc[locations['Well name'] == well_name]['Longitude'].values[0]
        df['lat'] = locations.loc[locations['Well name'] == well_name]['Latitude'].values[0]
        # df = df.loc[:, ['PERMEABILITY (HORIZONTAL)_Kair_md', 'POROSITY_(HELIUM)', 'NEUT', 'CALI','DENS', 'VpVsRatio', 'GR', 'PEF', 'RDEP', 'RMED', 'RSHAL', 'lon', 'lat', 'Facies Class Name']]

        dataframes.append(df)

    df = pd.concat(dataframes)
    df.dropna(subset = ['PERMEABILITY (HORIZONTAL)_Kair_md', 'POROSITY_(HELIUM)', 'Facies Class Name', 'DTS'],inplace=True)
    df = df.sample(frac=1)

    y = df['PERMEABILITY (HORIZONTAL)_Kair_md'].values
    X = df.drop('PERMEABILITY (HORIZONTAL)_Kair_md', axis=1)
    
    return X, y

def prepare_data(data_dir):
    wells, _ = well_name_list(data_dir)

    test_wells = ['204-20-1Z', '204-20-2', '204-20-6a'] # 10.75% of data
    val_wells = ['204-19-7'] # 10.43% of data
    train_wells = [well for well in wells if (well not in test_wells) & (well not in val_wells)] # 78.82% of data

    X_train, y_train = concat_data(data_dir, train_wells)
    X_val, y_val = concat_data(data_dir, val_wells)
    X_test, y_test = concat_data(data_dir, test_wells)

    return X_train, y_train, X_val, y_val, X_test, y_test