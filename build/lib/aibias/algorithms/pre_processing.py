import os
import sys

import numpy  as np
import pandas as pandas

import aibias.dataset as ds



#==================================================
#                   REWEIGHING
#==================================================


def Reweigh(dataset):
    """
    Adjusts the weights of each 'individual' such that those within the
    protected group with a positive and label and thouse without with a
    negative label are weighed higher, and those within the protected
    group with a negative lable and those without with a positive labbel
    are weighed lower

    Arguments:
        dataset - A dataset object

    Error:
        TypeError - If the given dataset is not of type
                    aibias.dataset.Dataset
    """

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.dataset.Dataset")
    
    df = dataset.dataframe.copy()


    num_pos = len(df[df['Label_binary']==1])
    num_neg = len(df) - num_pos
    num_tot = len(df)
    pr_pos  = num_pos / num_tot
    pr_neg  = num_neg / num_tot
    
    pre_computed = {
            0: pr_neg,
            1: pr_pos
        }

    weights = {
            0: dict(),
            1: dict()
        }

    for prot in [0,1]:      # Proteced vs Unproteced
        for lab in [0,1]:   # Favorable lable vs Unfavorable

            # Compute expected probability

            num_cur = len(df[df['Protected'] == prot])
            pr_cur  = num_cur / num_tot

            pr_lab = pre_computed[lab]

            pr_exp_cur = pr_lab * pr_cur


            # Compute observed probability

            num_obs_cur = len(df[(df['Protected']==prot)
                                &(df['Label_binary']==lab)])

            pr_obs_cur  = num_obs_cur / num_tot


            # Compute weight
            
            weight = pr_exp_cur / pr_obs_cur

            weights[prot][lab] = weight
            weights[prot]['pr_exp_cur || {}'.format(lab)] = pr_exp_cur
            weights[prot]['pr_obs_cur || {}'.format(lab)] = pr_obs_cur
            

    # Update weights
    
    for prot in [0,1]:
        for lab in [0,1]:
            df.loc[(df['Protected']==prot)
                  &(df['Label_binary']==lab),
                       'Weight'] = weights[prot][lab]

    # Create new dataset object with the transformed dataset
    
    transformed_dataset = ds.Dataset(
            df,
            label_names = dataset.label_names,
            protected_attribute_names = dataset.protected_attribute_names,
            title = dataset.title + ' (Reweighed)',
            weights = df['Weight'].values.copy(),
            alter_dataframe = False)

    return transformed_dataset




#===========================================
#       OPTIMIZED PRE-PROCESSING
#===========================================



class OptimizedPreProcessing():


    def __init__(self, dataset):

        if not isinstance(dataset,ds.Dataset):
            raise TypeError("Dataset must be of type aibias.dataset.Dataset")

        self.dataset = dataset


    def fit(self):
        print('TODO')
        return self


    def transform(self):
        print('TODO')


    def fit_transform(self):
        self.fit().transform()



