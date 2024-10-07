import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from skimage.transform import resize

import torch

def combine_columns(df):
    """
    Function to combine the columns of the dataframe into one column

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing the data to be combined

    Returns
    -------
    df : pandas dataframe
        Dataframe containing the combined data
    """
    # Collapse DTS_1 and DTS_2 to DTS1 and DTS2 respectively
    if "DTS_1" in df.columns and "DTS1" not in df.columns:
        df.rename(columns={"DTS_1": "DTS1"}, inplace=True)
    elif "DTS_1" in df.columns and "DTS1" in df.columns:
        df["DTS1"] = df["DTS1"].combine_first(df["DTS_1"])
    else:
        pass

    if "DTS_2" in df.columns and "DTS2" not in df.columns:
        df.rename(columns={"DTS_2": "DTS2"}, inplace=True)
    elif "DTS_2" in df.columns and "DTS2" in df.columns:
        df["DTS2"] = df["DTS2"].combine_first(df["DTS_2"])
    else:
        pass

    if "DTS1" in df.columns and "DTS" not in df.columns:
        df["DTS"] = df["DTS1"]
    elif "DTS1" in df.columns and "DTS" in df.columns:
        df["DTS"] = df["DTS"].combine_first(df["DTS1"])
    else:
        pass

    # Collapse RMIC and RMED to RMED
    if "RMIC" in df.columns and "RMED" not in df.columns:
        df.rename(columns={"RMIC": "RMED"}, inplace=True)
    elif "RMIC" in df.columns and "RMED" in df.columns:
        df["RMED"] = df["RMED"].combine_first(df["RMIC"])
    else:
        pass

    # Average the values of AT10 and AT20, and collapse it to RSHAL
    if "AT10" in df.columns and "AT20" in df.columns and "RSHAL" not in df.columns:
        df["RSHAL"] = (df["AT10"] + df["AT20"]) / 2
    elif "AT10" in df.columns and "AT20" in df.columns and "RSHAL" in df.columns:
        df["RSHAL"] = df["RSHAL"].combine_first((df["AT10"] + df["AT20"]) / 2)
    else:
        pass

    # Average the values of AT30 and AT60, and collapse it to RMED
    if "AT30" in df.columns and "AT60" in df.columns and "RMED" not in df.columns:
        df["RMED"] = (df["AT30"] + df["AT60"]) / 2
    elif "AT30" in df.columns and "AT60" in df.columns and "RMED" in df.columns:
        df["RMED"] = df["RMED"].combine_first((df["AT30"] + df["AT60"]) / 2)
    else:
        pass

    # Collapse AT90 to RDEP
    if "AT90" in df.columns and "RDEP" not in df.columns:
        df.rename(columns={"AT90": "RDEP"}, inplace=True)
    elif "AT90" in df.columns and "RDEP" in df.columns:
        df["RDEP"] = df["RDEP"].combine_first(df["AT90"])
    else:
        pass
    
    return df

def load_porosity(path):
    """
    Function to load the porosity data
    
    Parameters
    ----------
    path : str
        Path to the porosity data
    
    Returns
    -------
    por_df : pandas dataframe
        Dataframe containing the porosity data
    por_col : str
        Name of the porosity column
    depth_col : str
        Name of the depth column
    """

    por_df = pd.read_csv(path)

    # Search for the column names in por_df
    por_col = [col for col in por_df.columns if "por" in col.lower()][0]
    depth_col = [col for col in por_df.columns if "depth" in col.lower()][0]

    # Check if the depth and porosity columns are of float type
    if por_df[por_col].dtype != np.float64:
        try:
            por_df[por_col] = por_df[por_col].str.replace("L","1").str.replace(",", ".").astype(float)
        except ValueError as exc:
            raise ValueError("Porosity column cannot be converted to float type") from exc
    
    if por_df[depth_col].dtype != np.float64:
        try:
            por_df[depth_col] = por_df[depth_col].str.replace("L","1").str.replace(",", ".").astype(float)
        except ValueError as exc:
            raise ValueError("Depth column cannot be converted to float type") from exc

    # drop rows where porosity is NaN
    por_df = por_df.dropna(subset=[por_col])

    return por_df, por_col, depth_col

def calculate_porosity(depth_array, por_df, por_col, depth_col):
    """
    Function to calculate porosity from the porosity dataframe

    Parameters
    ----------
    depth_array : numpy array
        Array containing the depth values
    por_df : pandas dataframe
        Dataframe containing the porosity data
    por_col : str
        Name of the porosity column
    depth_col : str
        Name of the depth column

    Returns
    -------
    porosity : numpy array
        Array containing the porosity values
    """
    
    # Ensure the depth array is in ascending order
    if not np.all(np.diff(depth_array) >= 0):
        raise ValueError("Depth array must be sorted in ascending order")
    
    # Interpolate porosity values
    return np.interp(depth_array, por_df[depth_col].values, por_df[por_col].values, left=np.nan, right=np.nan)
    
def fill_random_permeability(df):
    """
    Function to fill the permeability column with random values
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing the data
    
    Returns
    -------
    permeability : pandas series
        Series containing the permeability values
    """
    grouped = df.groupby('well')
    permeability = pd.Series(index=df.index, dtype=np.float64, name="permeability")
    for well, group in grouped:
        # Generate random permeability values for the group
        group_permeability = np.random.uniform(1, 100, size=len(group))

        # Apply the logic for each depth in the group
        for i, (index, row) in enumerate(group.iterrows()):
            depth = row['DEPTH']
            core_depths = row['core_depths']
            if depth < core_depths[0] or depth > core_depths[1]:
                group_permeability[i] = np.nan

        # Assign the computed permeability values to the corresponding indices in the main series
        permeability[group.index] = group_permeability
    return permeability
    
def fill_random_facies(df):
    """
    Function to fill the facies column with random values
    
    Parameters
    ----------
    df : pandas dataframe
    
    Returns
    -------
    facies : pandas series
        Series containing the facies values
    """
    # Define your facies categories
    facies_categories = ["os", "s", "sh", "ms"]

    # Create an empty Series to store facies values
    facies = pd.Series(index=df.index, dtype="object", name="facies")

    # Group by well and process each group
    grouped = df.groupby('well')
    for well, group in grouped:
        # Generate random facies values for the group
        group_facies = np.random.choice(facies_categories, size=len(group))

        # Apply the logic for each depth in the group
        for i, row in group.iterrows():
            depth = row['DEPTH']
            core_depths = row['core_depths']
            if depth < core_depths[0] or depth > core_depths[1]:
                group_facies[i] = np.nan

        # Assign the computed facies values to the corresponding indices in the main series
        facies.loc[group.index] = group_facies

    return facies

def get_data_loader(data_path, data):
    """
    Function to query core images from the core image file.

    Parameters
    ----------
    data_path : str
        Path to core image folder
    data : pandas dataframe
        Dataframe containing well logs data

    Returns
    -------
    tensor_dataset : torch tensor
        contain well log and core image in tensor format
    """
    
    data = data.drop_duplicates()
    img_index = ['depth_file', 'image_file', 'DEPTH']

    depth_col = ['DEPTH']
    depth_transformer = StandardScaler()

    log_norm_cols = ['RDEP', 'RSHAL', 'RMED']
    i_transformer = make_pipeline(SimpleImputer(strategy='mean'), FunctionTransformer(np.log), RobustScaler())

    norm_cols = ['CALI', 'DENS', 'DTC', 'GR', 'NEUT', 'PEF', 'DTS1']
    h_transformer = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler())

    #combine splitted pipelines
    preprocessor_pipe = ColumnTransformer([
        ('depth', depth_transformer, depth_col),
        ('radiation', h_transformer, norm_cols),
        ('induction', i_transformer, log_norm_cols),
        ])

    feature_names = depth_col + norm_cols + log_norm_cols
    well_array_transformed = preprocessor_pipe.fit_transform(data)
    well_df_transformed = pd.DataFrame(well_array_transformed, columns=feature_names)

    X = well_df_transformed.astype(np.float32)

    tensor_X = torch.tensor(X.values, dtype=torch.float32)
    tensor_z = torch.tensor(get_all_images(data[img_index]), dtype=torch.float32)

    return tensor_z, tensor_X


def get_all_images(data):
        """
        Get all the images from the core image file

        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the core image file path and depth
            
        Returns
        -------
        image_list : list
            List containing the core images
        """
        image_list = [get_image(data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2]) for i in range(data.shape[0])]
        return np.stack(image_list, axis=0)

def get_image(depth_path, image_path, depth):
    """
    Function to get the core image from the core image file

    Parameters
    ----------
    depth_path : str
        Path to the depth file
    image_path : str
        Path to the core image file
    depth : float

    Returns
    -------
    image_resize : numpy array
        Array containing the core image
    """
    # If the file path exists, load the image. If not return a tensor of zeros (black image)
    if image_path is not None and depth_path is not None:
        #load the core image and depth
        core_image = np.load(image_path)
        core_image_depth = np.load(depth_path)
        
        # Get the start and end depth
        # depth range is 0.1524m above and below the depth (equal to 1 well log sample rate)
        start_depth_search = depth - 0.1524
        end_depth_search = depth + 0.1524

        # Define function to get depth index
        def get_depth_index(core_image_depth, min, max):
            array = np.where((core_image_depth >= min) & (core_image_depth <= max))
            return array[0][0], array[0][-1]
        
        # Get the start and end depth index
        start_depth, stop_depth = get_depth_index(core_image_depth, start_depth_search, end_depth_search)

        # Get the numpy array image from core image
        image = core_image[start_depth:stop_depth, :, :]
        
        # Resize image to 50x50 pixels
        image_resize = np.transpose(resize(image, (50, 50), anti_aliasing=True), (2, 0, 1))
        return image_resize
    else:
        # If the file path does not exist, return a tensor of zeros (black image)
        return np.zeros(3, 50, 50)