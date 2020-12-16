import numpy as np
import aibias.datasets.dataset as ds



#===================================================
#               DISPARATE IMPACT
#===================================================


def DisparateImpact(dataset):

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.datasets.dataset.Dataset")

    dataset.get_statistics()
    pr_prot   = dataset.Statistics['Protected']['Percentage']
    pr_unprot = dataset.Statistics['Unprotected']['Percentage']
    return pr_unprot / pr_prot



#===================================================
#       STATISTICAL PARITY DIFFERENCE          
#===================================================


def StatisticalParityDifference(dataset):

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.datasets.dataset.Dataset")

    dataset.get_statistics()
    pr_prot   = dataset.Statistics['Protected']['Percentage']
    pr_unprot = dataset.Statistics['Unprotected']['Percentage']
    return pr_prot - pr_unprot



#===================================================
#           AVERAGE ODDS DIFFERENCE
#===================================================


def AverageOddsDifference(dataset,predictions):
    
    dataset.get_statistics()

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.datasets.dataset.Dataset")
    if not isinstance(predictions,np.ndarray):
        raise TypeError("Predictions must of type np.ndarray")

    df = dataset.dataframe
    if not predictions.shape == (len(df),):
        raise ValueError("Predictions must be of shape (X,) where X is the size of the dataset")


    tp_prot   = 0
    fp_prot   = 0
    tp_unprot = 0
    fp_unprot = 0

    
    for i in range(len(df)):
        if df.loc[i]['Protected'] == 1:
            if df.loc[i]['Label_binary'] == 1:
                if predictions[i] == 1:
                    # True positive || Protected
                    tp_prot += 1
            else:
                if predictions[i] == 1:
                    # False positive || Protected
                    fp_prot += 1
        else:
            if df.loc[i]['Label_binary'] == 1:
                if predictions[i] == 1:
                    # True positive || Unprotected
                    tp_unprot += 1
            else:
                if predictions[i] == 1:
                    # False positive || Unprotected
                    fp_unprot += 1

    tpr_prot   = tp_prot   / dataset.Statistics['Protected']['Positive']
    fpr_prot   = fp_prot   / dataset.Statistics['Protected']['Positive']
    tpr_unprot = tp_unprot / dataset.Statistics['Unprotected']['Positive']
    fpr_unprot = fp_unprot / dataset.Statistics['Unprotected']['Positive']

    avg_prot   = (tp_prot   + fp_prot)   / 2
    avg_unprot = (tp_unprot + fp_unprot) / 2


    return avg_prot - avg_unprot



#===================================================
#           EQUAL OPPORTUNITY DIFFERENCE
#===================================================


def EqualOpportunityDifference(dataset,predictions):

    dataset.get_statistics()

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.datasets.dataset.Dataset")
    if not isinstance(predictions,np.ndarray):
        raise TypeError("Predictions must of type np.ndarray")

    df = dataset.dataframe
    if not predictions.shape == (len(df),):
        raise ValueError("Predictions must be of shape (X,) where X is the size of the dataset")


    tp_prot   = 0
    tp_unprot = 0


    for i in range(len(df)):
        if df.loc[i]['Protected'] == 1:
            if df.loc[i]['Label_binary'] == 1:
                if predictions[i] == 1:
                    # True positive || Protected
                    tp_prot += 1
        else:
            if df.loc[i]['Label_binary'] == 1:
                if predictions[i] == 1:
                    # True positive || Unprotected
                    tp_unprot += 1

    tpr_prot   = tp_prot   / dataset.Statistics['Protected']['Positive']
    tpr_unprot = tp_unprot / dataset.Statistics['Unprotected']['Positive']

    return tpr_prot - tpr_unprot
