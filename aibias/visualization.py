import os
import sys


import numpy  as np
import pandas as pd

from datetime import datetime

import aibias.dataset    as ds
import aibias.metrics    as met
import matplotlib.pyplot as plt


class Visualization():
    """
    A class that handles visualizations of metrics for datasets

    Attributes:
        datasets        - (list) List of datasets to be visualizes
        metrics         - (list) List of metrics implemented
        metrics_short   - (list) List of short hand names of metrics
        X               - (int) Size of x-axis of figure
        Y               - (int) Size of y-axis of figure
        save            - (boolean) To save the figures
        reference       - (str) 'label' or 'prediction', for determening
                          which to use in Disparate Impact and Statistical
                          Parity Difference

    Functions:
        visualize_metric  - Generates a figure with visualization of the
                            chosen metric for given datasets
        visualize_metrics - Generates figures for each metric implemented
                            for given datasets
    """

    def __init__(self, datasets, figsize_X=5, figsize_Y=5, 
            save=False,reference='label'):
        """
        Arguments:
            dataset     - (list) List of datasets to be visualized
            figsize_X   - (int) Size of the x-axis of figure
                          default: 5
            figsize_Y   - (int) Size of the y-axis of figure
                          default: 5
            save        - (boolen) To save the figure
                          default: False
            reference   - (str) 'label' or 'prediction', for determening
                          which to use in Disparate Impact and Statistical
                          Parity Difference
                          default: 'label'
        raises:
            TypeError: Datasets must be of type aibias.dataset.Dataset
        """
        
        if isinstance(datasets,list):
            self.datasets = datasets
        else:
            self.datasets = [datasets]
        
        for dataset in datasets:
            if not isinstance(dataset,ds.Dataset):
                raise TypeError("Datasets must be of type aibias.dataset.Dataset")

        self.metrics = [
                'DisparateImpact',
                'StatisticalParityDifference',
                'AverageOddsDifference',
                'EqualOpportunityDifference'
        ]
        self.metrics_short = ['di','spd','aod','eod']
        self.X = figsize_X
        self.Y = figsize_Y
        self.save = save
        self.reference = reference
        

    def visualize_metric(self,metric,rotation=90,annotation=True, 
            references = None, **kwargs):
        """
        Generates a figure with visualization of the chosen metric for
        given datasets

        Arguments:
            metric      - (str) Metric to visualize
            rotation    - (int) Rotation of bar labels
                          default: 90
            annotation  - (booleon) To annotate bars with values
                          default: True
            references  - (list) List of references ('label' or
                          'prediction') for when different datasets
                          use different references
                          default: None
        kwargs:
            All kwargs that could be used with pyplot.bar

        Raises:
            ValueError: metric must be one of the implemented metrics
            ValueError: Dataset must contain predictions
                        - For AverageOddsDifference and
                          EqualOpportunityDifference
        """

        if (not metric in self.metrics and 
            not metric.lower() in self.metrics_short):
            raise ValueError("`metric` must be one of the following: " +
                "{0[0]} (di), {0[1]} (spd),{0[2]} (aod), {0[3]} (eod)"
                .format(self.metrics))

        # Disparate Impact
        if metric.lower() == 'di' or metric == self.metrics[0]:

            values  = list()
            titles  = list()
            ylable  = self.metrics[0]

            for i,ds in enumerate(self.datasets):
                if not references is None:
                    di = met.DisparateImpact(ds,references[i])
                else:
                    di = met.DisparateImpact(ds,self.reference)
                values.append(di)
                titles.append(ds.title)

        # Statistical Parity Difference
        elif metric.lower() == 'spd' or metric == self.metrics[1]:

            values  = list()
            titles  = list()
            ylable  = self.metrics[1]

            for i,ds in enumerate(self.datasets):
                if not references is None:
                    spd = met.StatisticalParityDifference(ds,references[i])
                else: 
                    spd = met.StatisticalParityDifference(ds,self.reference)
                values.append(spd)
                titles.append(ds.title)

        # Average Odds Difference
        elif metric.lower() == 'aod' or metric == self.metrics[2]:

            values  = list()
            titles  = list()
            ylable  = self.metrics[2]

            for ds in self.datasets:

                if not 'Prediction' in ds.dataframe.columns:
                    raise ValueError("No predictions included in dataset")

                aod = met.AverageOddsDifference(ds)
                values.append(aod)
                titles.append(ds.title)

        # Equal Odds Difference
        elif metric.lower() == 'eod' or metric == self.metrics[3]:

            values  = list()
            titles  = list()
            ylable  = self.metrics[3]

            for ds in self.datasets:

                if not 'Prediction' in ds.dataframe.columns:
                    raise ValueError("No predictions included in dataset")

                eod = met.EqualOpportunityDifference(ds)
                values.append(eod)
                titles.append(ds.title)

            
        # Create figure
        fig = plt.figure(figsize=(self.X,self.Y))
        plt.gca().yaxis.grid()
        plt.gca().set_axisbelow(True)
        
        # Create bars
        kwargs.setdefault('color','cyan')
        kwargs.setdefault('edgecolor','black')
        kwargs.setdefault('alpha',0.5)
        plt.bar(titles,values,**kwargs)

        # Set fairness line
        if metric.lower() == 'di' or metric == self.metrics[0]:
            fair = 1
        else:
            fair = 0
        num_ds = len(self.datasets)
        plt.text(num_ds+0.5,fair,'Fair',fontsize=14)
        plt.hlines(fair,-1,num_ds,label='Fair',linewidth=2)

        # Set labels
        plt.ylabel(ylable)
        plt.xlabel('Datasets')
        plt.setp(plt.gca().get_xticklabels(), rotation=rotation, 
                    horizontalalignment='right')


        # Annotate bars
        if annotation:
            for i, val in enumerate(values):
                y_val   = max(0,val) + 0.05
                val     = np.round(val,decimals=4)
                if fair == 1 and abs(y_val-1) < 0.1:
                    y_val = 1.05
                plt.annotate(f'{val}\n',xy=(titles[i],y_val),
                             ha='center',va='center')


        # Set y axis range
        y_max = max(values+[0]) if fair == 0 else max(values+[1])
        y_min = min(values+[0])
        plt.ylim(y_min-.5,y_max+.5)


        # Save figure
        if self.save:
            plt.savefig('Dataset_graph_{}'.format(datetime.now()
                    .strftime('%y_%m_%d_%H:%M:%s')))

        plt.show()


    def visualize_metrics(self, rotation=90,
            references = None, **kwargs):
        """
        Generates figures for each metric implemented for given datasets

        Arguments:
            rotation    - (int) Rotation of bar labels
                          default: 90
            references  - (list) List of references ('label' or
                          'prediction') for when different datasets
                          use different references
                          default: None
        kwargs:
            All kwargs that could be used with pyplot.bar
        """

        for metric in self.metrics:
            self.visualize_metric(metric,rotation,
                    references=references, **kwargs)
