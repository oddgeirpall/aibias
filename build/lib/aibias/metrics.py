import numpy as np
import aibias.dataset as ds



#===================================================
#               DISPARATE IMPACT
#===================================================


def DisparateImpact(dataset,reference='label'):
    """
    The ratio in probability of favorable outcomes between unprivileged
    and privileged groups.

    DI = Pr(C = YES | X = 0) / Pr(C = YES | X = 1)

    Arguments:
        dataset - A dataset object

    Returns:
        DI - (float) Disparate Impact of dataset

    Error:
        TypeError - If the given dataset is not of type 
                    aibias.dataset.Dataset
    """

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.dataset.Dataset")

    dataset.get_statistics(reference)
    pr_prot   = dataset.Statistics['Protected']['Percentage']
    pr_unprot = dataset.Statistics['Unprotected']['Percentage']
    return pr_prot / pr_unprot



#===================================================
#       STATISTICAL PARITY DIFFERENCE          
#===================================================


def StatisticalParityDifference(dataset,reference='label'):
    """
    The difference in the probability of favorable outcomes between the
    unprivileged and privileged groups

    SPD = Pr(C = YES | X = 1) - Pr(C = YES | x = 0)

    Arguments:
        dataset - A dataset object

    Returns:
        SPD - (float) Statistical Parity Difference of dataset

    Error:
        TypeError - If the given dataset is not of type 
                    aibias.dataset.Dataset
    """

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.dataset.Dataset")

    dataset.get_statistics(reference)
    pr_prot   = dataset.Statistics['Protected']['Percentage']
    pr_unprot = dataset.Statistics['Unprotected']['Percentage']
    return pr_prot - pr_unprot



#===================================================
#           AVERAGE ODDS DIFFERENCE
#===================================================


def AverageOddsDifference(dataset):
    """
    The average of difference in flase positive rates and true positive
    rates between unprivileged and privilged groups

    AOD = AVG(TPR,FPR | X = 1) - AVG(TPR,FPR | X = 0)

    Arguments:
        dataset     - A dataset object
        predictions - An array containing the model predictions

    Returns:
        AOD - (float) Average Odds Difference of dataset

    Error:
        TypeError  - If the given dataset is not of type 
                     aibias.dataset.Dataset
        TypeError  - If the predictions are not of type np.ndarray
        ValueError - If the shape of predictions does not match the
                     size of the dataset
    """
    
    dataset.get_statistics()

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.dataset.Dataset")

    df = dataset.dataframe

    if not 'Prediction' in df.columns:
        raise ValueError("No predictions included in dataset")


    tp_prot   = df[ (df['Protected']==1)&(df['Label_binary']==1)&
                    (df['Prediction_binary']==1)]
    fp_prot   = df[ (df['Protected']==1)&(df['Label_binary']==0)&
                    (df['Prediction_binary']==1)]
    tp_unprot = df[ (df['Protected']==0)&(df['Label_binary']==1)&
                    (df['Prediction_binary']==1)]
    fp_unprot = df[ (df['Protected']==0)&(df['Label_binary']==0)&
                    (df['Prediction_binary']==1)]

    tpr_prot   = len(tp_prot)   / dataset.Statistics['Protected']['Positive']
    fpr_prot   = len(fp_prot)   / dataset.Statistics['Protected']['Positive']
    tpr_unprot = len(tp_unprot) / dataset.Statistics['Unprotected']['Positive']
    fpr_unprot = len(fp_unprot) / dataset.Statistics['Unprotected']['Positive']

    avg_prot   = (tpr_prot   + fpr_prot)   / 2
    avg_unprot = (tpr_unprot + fpr_unprot) / 2


    return avg_prot - avg_unprot



#===================================================
#           EQUAL OPPORTUNITY DIFFERENCE
#===================================================


def EqualOpportunityDifference(dataset):
    """
    The difference in true positive rates between unprivileged and
    privileged groups

    EOD = {TPR | Protected} - {TPR | Unprotected}

    Arguments:
        dataset     - A dataset object
        predictions - An array containing the model predictions

    Returns:
        EOD - (float) Equal Opoortunity Difference of dataset

    Error:
        TypeError  - If the given dataset is not of type 
                     aibias.dataset.Dataset
        TypeError  - If the predictions are not of type np.ndarray
        ValueError - If the shape of predictions does not match the
                     size of the dataset
    """

    dataset.get_statistics()

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.dataset.Dataset")

    df = dataset.dataframe

    if not 'Prediction' in df.columns:
        raise ValueError("No predictions included in dataset")


    tp_prot   = df[ (df['Protected']==1)&(df['Label_binary']==1)&
                    (df['Prediction_binary']==1)]
    tp_unprot = df[ (df['Protected']==0)&(df['Label_binary']==1)&
                    (df['Prediction_binary']==1)]

    tpr_prot   = len(tp_prot)   / dataset.Statistics['Protected']['Positive']
    tpr_unprot = len(tp_unprot) / dataset.Statistics['Unprotected']['Positive']

    return tpr_prot - tpr_unprot
