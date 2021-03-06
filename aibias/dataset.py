import os
import sys

import numpy  as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from datetime import datetime


class Dataset():
    """
    A class that handles datasets for the methods in this python package

    Attributes:
        dataframe       - (pandas.DataFrame) Dataframe containing dataset
        feature_names   - (list) Llist of feature names
        features        - (numpy.ndarray) Values of features in dataset
        labels          - (numpy.ndarrau) Values of real labels for dataset
        instance_names  - (list) List of indeces for dataset
        pred_name       - (str) Name of column containing predictions (optional)
        predictions     - (numpy.ndarray) Array containing predictions
                          from a classifier for dataset (optional)
        cat_features    - (list) List of features that are categorical (optional)
        train_features  - (list) List of features for training (optional)
        model           - (tensorflow model) A model trained using
                          the dataset (optional)
        title           - (str) Title of dataset (optional)

        protected_attribute_names   - (list) List of features that should
                                      be protected
        protected_attributes        - (numpy.ndarray) Array containing all
                                      values of protected attributes
        instance_weights            - (numpy.ndarray) Array containing
                                      weights for each row (optional)
        labels_binary               - (numpy.ndarray) Labels cast to
                                      zeros and ones

    Functions:
        __init__        - Initiates an instance of a Dataset
        get_statistics  - Updates statistic attributes in case of
                          change to dataset
    """

    def __init__(self, df, label_names, protected_attribute_names,
                 map_func_pa=None, map_func_lab=None, title=None,
                 weights = None, pred_name = None, predictions = None,
                 map_func_pred=None, categorical_features=None,
                 model = None, training_features=None, 
                 alter_dataframe=True):
        """
        Arguments:
            df              - Pandas dataframe containing features, labels
                              and protected attributes. All data should be
                              numerical (NAs not allowed).
            label_names     - (list) Names of the labels of the data
            map_func_pa     - Function that maps protected attributes
                              to 0 or 1
                              default: None
            map_func_lab    - Function that maps labels to 0 or 1
                              default: None
            title           - (str) Name of dataset
                              default: None
            weights         - (numpy.ndarray) Weights of each row
                              default: None
            pred_name       - (str) Column containing predictions from
                              a classifier for dataset
                              default: None
            predictions     - (numpy.ndarray) Array containing predictions
                              from a classifier on dataset
                              default: None
            map_func_pred   - Function that maps predictions to 0 or 1
                              default: None
            model           - A model trained using the dataset
                              default: None
            alter_dataframe - (boolean) Set to False if dataframe passed
                              to instance is already in the correct
                              format (mainly used by internal functions
                              within the package)
                              default: True

            protected_attribute_name    - (list) Names of the protected attributes
            training_features           - (list) List of features for training
                                          default: None
            categorical_features        - (list) List of features that are
                                          categorical
                                          default: None
                            

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

        
        self.dataframe = df
        self.feature_names = [n for n in df.columns if
                              n not in label_names]
        self.label_names    = label_names
        self.features       = df[self.feature_names].values.copy()
        self.labels         = df[self.label_names].values.copy()
        self.instance_names = df.index.astype(str).tolist()
        self.pred_name      = pred_name
        self.predictions    = predictions
        self.cat_features   = categorical_features
        self.train_features = training_features
        self.model          = model

        self.protected_attribute_names = protected_attribute_names
        self.protected_attributes      = (df.loc[:,protected_attribute_names]
                                         .values.copy())
        
        if not categorical_features is None:
            LE = LabelEncoder()
            for column in categorical_features:
                self.dataframe[column] = LE.fit_transform(
                                         self.dataframe[column])

        if self.pred_name:
            if map_func_pred:
                self.dataframe['Predictions']=map_func_pred(
                        self.dataframe[pred_name].values.copy()
                )
            else:
                self.dataframe['Predictions'] = (
                        self.dataframe[pred_name].values.copy()
                )
        elif not self.predictions is None:
            self.dataframe['Prediction'] = self.predictions
            self.dataframe.loc[self.dataframe['Prediction']>0.5,
                                          'Prediction_binary'] = 1
            self.dataframe.loc[self.dataframe['Prediction']<=0.5,
                                          'Prediction_binary'] = 0

        if weights is None:
            self.instance_weights = np.ones_like(self.instance_names,
                                             dtype=np.float64)
        else:
            self.instance_weights = weights

        if alter_dataframe:
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

        # Set dataset title
        if title:
            self.title = title
        else:
            self.title = 'Dataset_{}'.format(datetime.now()
                    .strftime('%y_%m_%d_%H:%M:%S'))

        # Basic statistics
        self.get_statistics()


    def get_statistics(self,reference='label'):
        """
        updates statistic attributes in case of change to dataset

        Args:
            reference - (str) Should be 'label' or 'prediction'
                        depending on which kind of statistics 
                        are desirable
                        default: 'label'

        Raises:
            ValueError: Reference must be either 'label' or 'prediction'
        """
    
        if reference == 'label':
            reference = 'Label_binary'
        elif reference == 'prediction':
            reference = 'Prediction_binary'
        else:
            raise ValueError("Reference must be either 'label' or 'prediction'")

        df = self.dataframe
        num_prot   = len(df[df['Protected'] == 1])
        num_unprot = len(df[df['Protected'] == 0])

        pos_prot   = sum(df[(df['Protected']==1)&
                        (df[reference]==1)]['Weight'])
        pos_unprot = sum(df[(df['Protected']==0)&
                        (df[reference]==1)]['Weight'])

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

