import os
import sys

import numpy  as np
import pandas as pd

import aibias.dataset as ds



#=============================================
#        REJECT OPTION CLASSIFICATION
#=============================================


def RejectOption(dataset,Theta):
    """
    Takes in a dataset with non-discrete predictions by a classifier
    and changes results within the critical area such that protected
    individuals all receive a favorable label and non-protected
    individuals all receive an unfavorable label. Base on the paper

    'Decision Theory for Discrimination-aware Classification'
    Faisal Kamiran et al. (2012)

    Critical area = [0.5-Theta, 0.5+Theta], Theta in [0,0.5]

    Arguemnts:
        dataset - (aibias.dataset.Dataset) Dataset object
        Theta   - (float) Determines the size of the critical area

    Returns:
        New dataset object with transformed predictions

    Raises:
        TypeError: Dataset must be of type aibias.dataset.Dataset
        ValueError: Dataset must contain predictions
    """

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.dataset.Dataset")

    df = dataset.dataframe.copy()

    if 'Prediction' not in df.columns:
        raise ValueError("Dataset must contain predictions")

    mask = abs(df['Prediction']-0.5)<Theta
    df.loc[(df['Protected']==1)&mask,'Prediction_binary'] = 1
    df.loc[(df['Protected']==0)&mask,'Prediction_binary'] = 0


    # Create new dataset object with the transformed dataset
    
    transformed_dataset = ds.Dataset(
            df, 
            label_names = dataset.label_names,
            protected_attribute_names = dataset.protected_attribute_names,
            title = dataset.title + ' (RejectOption)',
            alter_dataframe = False)

    return transformed_dataset
