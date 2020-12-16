import os
import sys


import numpy  as np
import pandas as pd

from datetime import datetime

import aibias.dataset    as ds
import aibias.metrics    as met
import matplotlib.pyplot as plt


class Visualization():

    def __init__(self, datasets, figsize_X=5, figsize_Y=5, 
            save=False,reference='label'):
        
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
        

    def visualize_metric(self,metric,rotation=90,annotation=True, **kwargs):

        if (not metric in self.metrics and 
            not metric.lower() in self.metrics_short):
            raise ValueError("`metric` must be one of the following: " +
                "{0[0]} (di), {0[1]} (spd),{0[2]} (aod), {0[3]} (eod)"
                .format(self.metrics))
        
        if metric.lower() == 'di' or metric == self.metrics[0]:

            values  = list()
            titles  = list()
            ylable  = self.metrics[0]

            for ds in self.datasets:
                di = met.DisparateImpact(ds,self.reference)
                values.append(di)
                titles.append(ds.title)

        elif metric.lower() == 'spd' or metric == self.metrics[1]:

            values  = list()
            titles  = list()
            ylable  = self.metrics[1]

            for ds in self.datasets:
                spd = met.StatisticalParityDifference(ds,self.reference)
                values.append(spd)
                titles.append(ds.title)

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


    def visualize_metrics(self, rotation=90, **kwargs):

        for metric in self.metrics:
            self.visualize_metric(metric,rotation,**kwargs)

