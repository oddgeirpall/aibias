import os
import sys

import numpy  as np
import pandas as pd




class Dataset():
    """
    A class to handle datasets
    """

    def __init__(self, df, label_names, protected_attribute_names,
                 map_func_pa=None, map_func_lab=None):
        """
        Arguments:
            df              - Pandas dataframe containing features, labels
                              and protected attributes. All data should be
                              numerical (NAs not allowed).
            labels          - Names of the labels of the data
            protected_attribute_name 
                            - Names of the protected attributes
            map_func_pa     - Function that maps protected attributes
                              to 0 or 1
            map_func_lab    - Function that maps labels to 0 or 1


        Raises:
            - TypeError:  Data must be a pandas dataframe
            - TypeError:  Certain fields must be np.ndarrays
            - ValueError: np.ndarray shapes must match
        """

        if not isinstance(df,pd.DataFrame):
            raise TypeError("Data must be provided as a pandas dataframe")
        if df is None:
            raise TypeError("Data not presented. Must provide a pandas "
                            "DataFrame with features, labels and a "
                            "protected attribute.")
        if df.isna().any().any():
            raise ValueError("DataFrame cannot contain any NA values.")

        # Make sure column, label and feature names are strings
        #df.columns               = df.columns.astype(str).tolist()
        #label_names              = list(map(str,label_names))
        #protected_attribute_name = list(map(str,protected_attribute_names))

        
        self.dataframe = df
        self.feature_names = [n for n in df.columns if
                              n not in label_names]
        self.label_names = label_names
        self.features = df[self.feature_names].values.copy()
        self.labels = df[self.label_names].values.copy()
        self.instance_names = df.index.astype(str).tolist()

        self.protected_attribute_names = protected_attribute_names
        self.protected_attributes      = (df.loc[:,protected_attribute_names]
                                         .values.copy())

        self.instance_weights = np.ones_like(self.instance_names,
                                             dtype=np.float64)
        # Map protected attributes and labels to 0//1
        if map_func_pa:
            self.protected_attributes_binary = map_func_pa(
                    self.protected_attributes
            )
        else:
            self.protected_attributes_binary = self.protected_attributes
        if map_func_lab:
            self.labels_binary =  map_func_lab(self.labels)
        else:
            self.labels_binary = self.labels

        self.dataframe['Protected']    = self.protected_attributes_binary
        self.dataframe['Label_binary'] = self.labels_binary
        self.dataframe['Weight']       = self.instance_weights


        # Basic statistics

        self.get_statistics()


    def get_statistics(self):
    
        df = self.dataframe
        num_prot   = len(df[df['Protected'] == 1])
        num_unprot = len(df[df['Protected'] == 0])

        pos_prot   = sum(df[(df['Protected']==1)&(df['Label_binary']==1)]['Weight'])
        pos_unprot = sum(df[(df['Protected']==0)&(df['Label_binary']==1)]['Weight'])

        pr_prot   = pos_prot   / num_prot
        pr_unprot = pos_unprot / num_unprot

        self.Statistics = {
                    'Protected': {
                        'Number'     : num_prot,
                        'Positive'   : pos_prot,
                        'Percentage' : pr_prot
                        },
                    'Unprotected': {
                        'Number'     : num_unprot,
                        'Positive'   : pos_unprot,
                        'Percentage' : pr_unprot
                        }
                 }

