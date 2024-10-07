import os
import pandas as pd
import numpy as np

from skimage.transform import resize

import torch
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin

class VpVsRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['VpVsRatio'] = X_transformed['DTC'] / X_transformed['DTS']
        X_transformed.drop(['DTC', 'DTS'], axis=1, inplace=True)
        return X_transformed

class custom_imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.to_log_features = ['CALI','GR', 'PEF', 'NEUT', 'DENS']
        self.no_log_features = ['POROSITY_(HELIUM)', 'SP']
        self.VpVs_features = ['DTC', 'DTS']

        self.num_features = self.to_log_features + self.no_log_features + self.VpVs_features
    
        self.group_medians_= []
        for i, feature in enumerate(self.num_features):
            self.group_medians_.append(X.groupby('Facies Class Name')[feature].median())
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for i, feature in enumerate(self.num_features):
            for category, median in self.group_medians_[i].items():
                X_transformed.loc[X_transformed['Facies Class Name'] == category, feature] \
                    = X_transformed.loc[X_transformed['Facies Class Name'] == category, feature].fillna(median)
            
        return X_transformed


class SP_imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby('Facies Class Name')['SP'].median()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.dropna(subset=['Facies Class Name'], inplace=True)
        for category, median in self.group_medians_.items():
            X_transformed.loc[X_transformed['Facies Class Name'] == category, 'SP'] \
                = X_transformed.loc[X_transformed['Facies Class Name'] == category, 'SP'].fillna(median)
            
        return X_transformed

class SP_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        # Transform both 'is' and 'ih'in Facies Class Name to combined class 'ms' (messy shale)
        X_transformed.loc[X_transformed['Facies Class Name'].isin(['is', 'ih']), 'Facies Class Name'] = 'ms'
        X_transformed.dropna(subset=['Facies Class Name'], inplace=True)

        return X_transformed