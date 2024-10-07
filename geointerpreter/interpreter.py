import os

import numpy as np
import pandas as pd

import lasio

from .extras.utility import combine_columns, load_porosity, calculate_porosity, \
    fill_random_permeability, fill_random_facies
from .extras.loaders import ModelLoaders
from .extras.utility import get_data_loader
from .visualization import facies_plot

_data_dir = os.path.join(os.path.dirname(__file__), 'data')

facies_classes_methods = {
    "RandomClass": "Facies Classification - Random Class Assignment",
    "ConcatCNN": "Facies Classification - ConcatCNN Model"
}

permeability_methods = {
    "RandomValue": "Permeability Prediction - Random Value Assignment",
    "RandomForest": "Permeability Prediction - Random Fores Model",
    "SVR": "Permeability Prediction - Support Vector Regression Model",
    "XGB": "Permeability Prediction - XGBoost Model"
}

facies_dict = {
    0: 'ms', 1: 'os', 2: 's', 3: 'sh'
}

class Interpreter():
    """
    Class to predict permeability and facies with trained machine learning
    and deep learning models from raw log and processed core image data.
    """

    def __init__(self, data_folder=""):
        """
        Parameters
        ----------

        data_folder : str, optional
            Path to folder containing the data files. The folder must contain
            the following files:
                - .las: raw log data file
                - .npy: processed core image data file
                - _depths.npy: array containing the depths of the processed core
                - _loc.csv: csv file containing the location of well 
        """

        if data_folder:
            self.data_folder = data_folder
        else:
            self.data_folder = _data_dir
        
        self.df = self._prepare_data()

        self.models = {}

        
    def _prepare_data(self):
        """
        Function to prepare data for prediction

        returns
        -------
        df : pandas dataframe
            Dataframe containing the data for prediction
        """
        # List out the wells in data folder by searching for .las files
        wells = [f.split("_")[0] for f in os.listdir(self.data_folder) if f.endswith('.las')]

        # Search for each well's .las file and .npy files. and put data into a dataframe
        _df = pd.DataFrame()

        for well in wells:
            print(f"Loading {well}...")
            las_path = os.path.join(self.data_folder, well) + '_logs.las'
            _temp_las = lasio.read(las_path)
            _temp_df = _temp_las.df().reset_index()
            _temp_df["well"] = well
            _temp_df["lon"] = _temp_las.well["LONG"].value
            _temp_df["lat"] = _temp_las.well["LATI"].value
            
            # Add the .npy file path as new column
            for file in ["depth", "image"]:
                _file_path = os.path.join(self.data_folder, well + f'_{file}.npy')
                _temp_df[f'{file}_file'] = _file_path
                if file == "depth":
                    depth_range = np.load(_file_path).min(), np.load(_file_path).max()
                    _temp_df["core_depths"] = [depth_range] * len(_temp_df)
            # drop rows above and below the core depth range
            _temp_df = _temp_df[_temp_df["DEPTH"].between(_temp_df["core_depths"].apply(lambda x: x[0]), 
                                                          _temp_df["core_depths"].apply(lambda x: x[1]))]

            
            # Porosity loader workflow
            _por_file = os.path.join(self.data_folder, well + ".csv")
            _por_df, _por_col, _depth_col = load_porosity(_por_file)
            _temp_df["porosity"] = calculate_porosity(_temp_df["DEPTH"].values, _por_df, _por_col, _depth_col)
            
            _df = pd.concat([_df, _temp_df], ignore_index=True)

        # Condition the dataframe columns to combine same meaning columns into one
        _df = combine_columns(_df)

        return _df
    
    def load_models(self, models=None, verbose=True):
        """
        Function to load the trained models for prediction

        Parameters
        ----------
        models : list, optional
            List of models to load. Default is None
        verbose : bool, optional
            Whether to print the loading status. Default is True
        

        Examples
        --------
        >>> interpreter = Interpreter()
        >>> fc_methods = list(facies_classes_methods.keys())
        >>> interpreter.load_models(fc_methods[0])
        >>> classes = interpreter.predict_facies(fc_methods[0])
        """
        if models is None:
            models = []
            
        for model in models:
            if verbose:
                print(f"Loading {model}...")
            self.models[model] = ModelLoaders().load_model(model)

    def predict_permeability(self, method="RandomValue", facies_method="RandomClass"):
        """ 
        Function to predict permeability using the trained models
        
        Parameters
        ----------
        method : str, optional
            Method to use for prediction. Default is RandomForest
        facies_method : str, optional
            Method to use for predicting facies. Default is ConcatCNN
        
        Returns
        -------
        permeability : pandas series
            Series containing the predicted permeability data
        """
        if method == "RandomValue":
            return fill_random_permeability(self.df)

        if method not in self.models:
            self.load_models([method])
        
        # Create X from df and predicted facies data
        if "facies" not in self.df.columns:
            if facies_method not in self.models:
                self.load_models([facies_method])
            facies = self.predict_facies(facies_method)
            X = pd.concat([self.df, facies], axis=1).copy()
        else:
            X = self.df.copy()

        X.rename(columns={"facies": "Facies Class Name",
                          "porosity": "POROSITY_(HELIUM)"}, inplace=True)
            
        return pd.Series(self.models[method].predict(X),
                            index=self.df.index,
                            name="permeability")

    def predict_facies(self, method="RandomClass"):
        """
        Function to predict facies using the trained models
        
        Parameters
        ----------
        method : str, optional
            Method to use for prediction. Default is RandomForest

        Returns
        -------
        facies : pandas series
            Series containing the predicted facies data
        """

        if method == "RandomClass":
            return fill_random_facies(self.df)
        
        if method not in self.models:
            self.load_models([method])

        # Create X from df
        X = self.df.copy()
        img, welllog = get_data_loader(self.data_folder, X)
        facies_probability = self.models[method](img, welllog).detach().numpy()
        facies_array = [facies_dict[index] for index in np.argmax(facies_probability, axis=1)]
        return pd.Series(facies_array,
                            index=self.df.index,
                            name="facies")
    
    def update_df(self, return_values=False, permeability_method="RandomValue", facies_method="RandomClass"):
        """
        Function to update the dataframe with the predicted permeability and facies

        Parameters
        ----------
        permeability_method : str, optional
            Method to use for predicting permeability. Default is RandomForest
        facies_method : str, optional
            Method to use for predicting facies. Default is ConcatCNN
        """
        if "facies" in self.df.columns and "permeability" in self.df.columns:
            print("Dataframe already contains the predicted permeability and facies")
        else:       
            if "facies" not in self.df.columns:
                self.df["facies"] = self.predict_facies(facies_method)
                print("Dataframe updated with predicted facies")

            if "permeability" not in self.df.columns:
                self.df["permeability"] = self.predict_permeability(permeability_method, facies_method)
                print("Dataframe updated with predicted permeability")
            
        if return_values:
            return self.df
    
    def save_df(self, path=""):
        """
        Function to save the dataframe to csv file
        
        Parameters
        ----------
        path : str, optional
            Path to save the dataframe. Default is current working directory
        """
        if path:
            self.df.to_csv(path, index=False)
        else:
            self.df.to_csv("predictions.csv", index=False)
        
    def plot_results(self):
        """
        Function to plot the predicted permeability and facies
        """
        self.update_df()
        facies_plot(self.df)
        


        